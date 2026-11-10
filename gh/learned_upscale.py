"""Learned AV-latent upscale for MiniMax H3 — LK edition (dense transfer).

结案陈词（同心圆涟漪案）：检查点的解码器对特征相位网格有固定约定——
普通 trilinear 插值在非训练倍率上会把特征"劈裂相位"（T8 dense_patch_grid
注释原话 "rather than splitting phases"），解码器看到相位错排的输入就输出
同心环伪影。T8 的连续坐标方案证明：同一份权重，只要把"编码器→解码器的
空间映射"换成相位校正的 grid_sample 栅格，就能在任意倍率上工作
（H05 已审 1.2×，连续倍率为其设计语义）。

本节点移植该推理路径（dense 前向）：
1. conv_in + in_blocks（嵌入 = 实际几何倍率 - 1.0）；
2. resize_dense_field：相位校正栅格逐帧 grid_sample（时间维原样保留）；
3. out_blocks + norm_out/silu + conv_out。
目标 latent = 一跑 latent × scale_by（取偶 → 画布自动 /32），画幅比例基本
保持（各轴偶数取整的偏差量级见入口注释）；精修继续工作在熟悉的目标画布上。

时间维默认按 lk_layout 分镜边界分批前向（每窗至多 8 帧上下文头，输出侧
裁掉），显存峰值 O(单窗)，长片不随分镜数上涨；`chunk_mode="whole"` 保留
整片一次的行为作 A/B 对照。GroupNorm 统计随分批内化为"每窗一份"——
检查点按 ≤362 实帧（~15s）的单条 clip 训练：整片拼接必越界，超过 clip
长度的单镜同样越界，逐镜分批是相对更贴训练分布的取法。

检查点契约、归一化常数与网络架构移植自 T8 参考实现，权重无需任何转换。
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .core import nested_av_parts
from .latent_upscale import _upscale_video_chunk

# Per-channel latent normalization the checkpoints were trained against
# (ported verbatim from the T8 reference implementation).
LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)

PRECISIONS = ("fp16", "bf16", "fp32")
RELEASE_POLICIES = ("offload_after", "clear_after", "keep_loaded")


def _precision_dtype(precision: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    try:
        return mapping[precision]
    except KeyError as exc:
        raise ValueError(f"Unknown precision {precision!r}; expected one of {PRECISIONS}") from exc


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int = 512, embed_dim: int = 64):
        super().__init__()
        self.in_layers = nn.Sequential(
            _group_norm(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, channels * 2))
        self.out_norm = _group_norm(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_layers(x)
        scale, shift = self.emb_layers(embedding).chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        hidden = self.out_norm(hidden) * (1.0 + scale) + shift
        return x + self.out_layers(hidden)


class _TemporalConv(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.norm = _group_norm(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class MiniMaxH3LearnedResizer3D(nn.Module):
    """Network reconstructed from the published 3D checkpoint tensor contract."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(24, 512, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.in_blocks = self._make_blocks()
        self.out_blocks = self._make_blocks()
        self.norm_out = _group_norm(512)
        self.conv_out = nn.Conv3d(512, 24, 3, padding=1)

    @staticmethod
    def _make_blocks() -> nn.ModuleList:
        blocks: list[nn.Module] = []
        for index in range(12):
            blocks.append(_ResBlockEmb3D())
            if index % 2 == 0:
                blocks.append(_TemporalConv())
        return nn.ModuleList(blocks)

    @staticmethod
    def _run_blocks(hidden, embedding, blocks) -> torch.Tensor:
        for block in blocks:
            if isinstance(block, _ResBlockEmb3D):
                hidden = block(hidden, embedding)
            else:
                hidden = block(hidden)
        return hidden


def _expected_state_contract() -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        model = MiniMaxH3LearnedResizer3D()
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


EXPECTED_STATE_CONTRACT = _expected_state_contract()


@dataclass
class _CachedModel:
    model: MiniMaxH3LearnedResizer3D
    mtime: float
    size: int


_CACHE: dict[tuple[str, float, int], _CachedModel] = {}


def _load_model(model_name: str, precision: str) -> _CachedModel:
    path = Path(folder_paths.get_full_path_or_raise("latent_upscale_models", model_name))
    key = (str(path), precision, path.stat().st_mtime, path.stat().st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    state = comfy.utils.load_torch_file(str(path))
    contract = EXPECTED_STATE_CONTRACT
    if len(state) != len(contract) or set(state) != set(contract):
        raise ValueError(
            f"{model_name} 不是本节点支持的 24 通道学习型重采样检查点 "
            f"(keys={len(state)}, expected={len(contract)})"
        )
    model = MiniMaxH3LearnedResizer3D()
    model.load_state_dict(state)
    del state
    model.to(_precision_dtype(precision))
    model.eval()
    entry = _CachedModel(model=model, mtime=key[2], size=key[3])
    _CACHE.clear()  # 单槽缓存：换检查点/精度即淘汰旧权重
    _CACHE[key] = entry
    return entry


def _register_model_folder() -> None:
    # Independence from the T8 pack: make sure the folder key exists even if
    # that pack is disabled.  Harmless no-op when it is already registered.
    try:
        if folder_paths.get_filename_list("latent_upscale_models"):
            return
    except Exception:
        pass
    base = os.path.join(folder_paths.models_dir, "latent_upscale_models")
    os.makedirs(base, exist_ok=True)
    try:
        folder_paths.add_folder_path("latent_upscale_models", base, False)
    except Exception as exc:
        logging.warning("[LKH3] latent_upscale_models 目录注册失败（可能已由其他包注册）: %s", exc)


_register_model_folder()


def _upscale_model_options() -> list[str]:
    """Safe schema options: an unregistered folder key must not break pack load."""
    try:
        names = folder_paths.get_filename_list("latent_upscale_models")
        if names:
            return list(names)
    except Exception:
        pass
    return ["minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"]


SCRATCH_BYTES = 64 << 20  # 逐帧栅格重采样的缓冲预算（移植自 T8 dense_transfer_exp）


def _dense_patch_grid(source_hw, target_hw, *, device):
    """目标 dense 格心在源 grid_sample 坐标系中的位置（检查点 patch 相位对齐）。"""
    sh, sw = source_hw
    th, tw = target_hw
    source_area = math.sqrt(sh * sw)
    target_area = math.sqrt(th * tw)
    grid_axes = []
    for source_length, target_length in ((sw, tw), (sh, th)):
        # Core positions one 2x2 patch at 16*(1-n/sqrt(HW)) + p*64/sqrt(HW).
        # Dense centers straddle that patch center, rather than splitting phases.
        source_spacing = 32.0 / source_area
        source_origin = 16.0 * (1.0 - source_length / source_area) - source_spacing / 2.0
        target_spacing = 32.0 / target_area
        target_origin = 16.0 * (1.0 - target_length / target_area) - target_spacing / 2.0
        coordinates = torch.arange(target_length, device=device, dtype=torch.float32)
        coordinates = coordinates * target_spacing + target_origin
        source_indices = (coordinates - source_origin) / source_spacing
        grid_axes.append(source_indices * 2.0 / (source_length - 1) - 1.0)
    x_axis, y_axis = grid_axes
    xx = x_axis.unsqueeze(0).expand(th, tw)
    yy = y_axis.unsqueeze(1).expand(th, tw)
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def _resize_dense_field(value, target_h, target_w):
    """空间连续传输（相位保持），缓冲有界，B/C/T 不变。"""
    if value.ndim != 5 or not value.is_floating_point():
        raise TypeError("dense transport requires floating BxCxTxHxW features")
    sh, sw = value.shape[-2:]
    if (sh, sw) == (target_h, target_w):
        return value
    b, c, t = value.shape[:3]
    grid = _dense_patch_grid((sh, sw), (target_h, target_w), device=value.device)
    # 逐帧独立处理；网络本身从不分块。
    destination = value.new_empty((b, c, t, target_h, target_w))
    chunk_frames = max(1, SCRATCH_BYTES // (c * (sh * sw + target_h * target_w) * 4))
    for batch in range(b):
        for first in range(0, t, chunk_frames):
            last = min(t, first + chunk_frames)
            frames = value[batch, :, first:last].movedim(1, 0).float()
            resized = F.grid_sample(
                frames, grid.expand(last - first, -1, -1, -1),
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            destination[batch, :, first:last] = resized.movedim(0, 1).to(value.dtype)
    return destination


def _dense_network_forward(model, normalized, effective_scale, target_h, target_w):
    """复用检查点权重/层，只替换编码器→解码器的空间映射（相位校正栅格）。"""
    embedding = model.embed(normalized.new_tensor([[effective_scale - 1.0]]))
    embedding = embedding.expand(normalized.shape[0], -1)
    features = model._run_blocks(model.conv_in(normalized), embedding, model.in_blocks)
    features = _resize_dense_field(features, target_h, target_w)
    features = model._run_blocks(features, embedding, model.out_blocks)
    return model.conv_out(F.silu(model.norm_out(features)))


CHUNK_MODES = ("segments", "fixed", "whole")
DEFAULT_CHUNK_WINDOW = 96  # ≈324 实帧，仍在检查点训练 clip 长度（124–362 实帧）之内
# 每窗喂入窗首之前至多这么多帧真实内容（输出侧裁掉），消化 (5,1,1) 时序卷积
# 栈在零填充边界上的污染——VAE 时序协议同样用 overlap 上下文而非裸切。
_CONTEXT_FRAMES = 8
_CHUNK_MODE_LABELS = {"segments": "按分镜", "fixed": "固定窗口", "whole": "整片"}


def _forward_chunk_plan(total_t, layout, mode, window):
    """潜空间时间分批计划：返回 (spans, effective_mode)。

    spans 为 [start, end) 半开区间列表，互不重叠且恰好覆盖 [0, total_t)；
    effective_mode 记录实际生效的计划方式（segments 校验失败回退 fixed 时
    与请求不同）。

    卷积网络对任意 T 切片都合法（此处不做 VAE 解码，无 5n+2 网格约束），
    逐窗前向后 cat 回 dim=2 即可。网络唯一的跨窗耦合是 GroupNorm 在整个
    (T,H,W) 上算统计量：切块后统计变"每窗一份"——整片拼接本就是超长越界
    输入（检查点按 ≤362 实帧的单条 clip 训练），每镜统计反而更贴训练分布，
    且切缝全部落在硬切镜头边界上，视觉上被剪切本身掩盖。前向侧每窗另喂
    窗首前至多 _CONTEXT_FRAMES 帧真实上下文（输出裁掉），减轻时序卷积的
    零填充边界污染（理论感受野单边 74 token，上下文只覆盖近端；随机初始化
    权重实测全局余弦 ~0.99、近缝逐帧余弦 ≥0.8，真实检查点漂移更小——
    见 tests/test_learned_upscale.py）。
    """
    total_t = int(total_t)
    if total_t <= 0:
        raise ValueError(f"潜空间时间长度为 {total_t}，无法分批放大")
    window = max(8, int(window))
    if mode == "whole":
        return [(0, total_t)], "whole"
    if mode == "segments":
        spans = []
        try:
            for seg in (layout or {}).get("segments") or []:
                start = int(seg["video_offset"])
                tokens = int(seg["video_tokens"])
                if start < 0 or tokens <= 0:
                    spans = []
                    break
                spans.append((start, start + tokens))
        except (TypeError, AttributeError, ValueError, KeyError, OverflowError):
            spans = []  # segments 结构/数值异常（非 dict、缺键、inf 等）→ 当漂移处理
        if (spans and spans[0][0] == 0
                and all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
                and spans[-1][1] == total_t):
            return spans, "segments"
        # 布局漂移（过期 layout / 契约模板透传 / 非本包 latent）：回退固定窗口，
        # 别让长片炸显存。回退事实记入 report["chunks"]["effective_mode"/"fallback"]。
        logging.warning(
            "[LKH3] 分批放大：lk_layout 与潜空间长度不一致（T=%d），回退固定窗口 %d",
            total_t, window,
        )
    return [(first, min(first + window, total_t)) for first in range(0, total_t, window)], "fixed"


def learned_upscale_h3_av_latent(
    *,
    av_latent: dict,
    model_name: str,
    scale_by: float,
    precision: str,
    release_policy: str,
    chunk_mode: str = "segments",
    chunk_window: int = DEFAULT_CHUNK_WINDOW,
):
    video, audio = nested_av_parts(av_latent)
    # lk_layout 只做只读透传 + video_hw 更新：非 dict（手改/外来 latent）按空
    # 布局处理，与 _forward_chunk_plan 的容错同口径——绝不在跑完整片前向之后
    # 才因 dict(非dict) 崩溃。
    raw_layout = av_latent.get("lk_layout")
    layout = dict(raw_layout) if isinstance(raw_layout, dict) else {}
    src_h, src_w = int(video.shape[3]), int(video.shape[4])
    # H3 条件化画布必须 /32（32px = 2 个 latent token）。NaN/Inf 的 scale_by
    # 若走钳位会静默变 noop——镜像二跑精修的非有限数值防御：告警 + 回退默认。
    scale_by_f = float(scale_by)
    if not math.isfinite(scale_by_f):
        logging.warning(
            "[LKH3] 学习型放大：scale_by 非有限数值（%r），回退默认 1.5", scale_by,
        )
        scale_by_f = 1.5
    scale = min(4.0, max(1.0, scale_by_f))
    # 目标 latent = 一跑 latent × scale，各轴独立取偶（画布自动 /32）；round 的
    # banker's rounding 会在 .5 边界给出各向异性最差的组合（46x26 @1.5 →
    # (68,40)，两轴倍率差 4%），改为在 2x2 偶数邻域里选两轴倍率最接近的组合
    # ——大画布典型各向异性 <2%，小/极端画幅可超。
    base_h, base_w = src_h * scale / 2.0, src_w * scale / 2.0
    best = None
    for h in (max(2, math.floor(base_h)) * 2, max(2, math.ceil(base_h)) * 2):
        for w in (max(2, math.floor(base_w)) * 2, max(2, math.ceil(base_w)) * 2):
            anisotropy = abs((h / src_h) / (w / src_w) - 1.0)
            if best is None or anisotropy < best[0]:
                best = (anisotropy, h, w)
    tgt_latent_h, tgt_latent_w = best[1], best[2]
    tgt_w, tgt_h = tgt_latent_w * 16, tgt_latent_h * 16

    if (src_h, src_w) == (tgt_latent_h, tgt_latent_w):
        report = {
            "schema_version": 1, "node": "LKH3AVLearnedUpscale", "status": "noop",
            "source_latent": [src_h, src_w], "target_px": [tgt_w, tgt_h],
        }
        out = dict(av_latent)
        out["lk_layout"] = dict(layout)
        out["lk_layout"]["video_hw"] = [tgt_latent_h, tgt_latent_w]
        return out, tgt_w, tgt_h, json.dumps(report, ensure_ascii=False, sort_keys=True)

    # dense 嵌入喂实际几何倍率（两轴比例的几何均值）——连续坐标检查点的训练语义。
    effective_scale = math.sqrt((tgt_latent_h / src_h) * (tgt_latent_w / src_w))

    # 计划先行：T<=0 / 手改参数在模型加载前就报错，不白付一次权重加载。
    spans, effective_mode = _forward_chunk_plan(
        int(video.shape[2]), layout, chunk_mode, chunk_window,
    )
    entry = _load_model(model_name, precision)
    device = comfy.model_management.get_torch_device()
    dtype = _precision_dtype(precision)
    on_device = all(p.device == device for p in entry.model.parameters())
    on_dtype = all(p.dtype == dtype for p in entry.model.parameters())
    if not (on_device and on_dtype):
        entry.model.to(device=device, dtype=dtype)
    mean = torch.tensor(LATENTS_MEAN, device=device, dtype=dtype).view(1, 24, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device=device, dtype=dtype).view(1, 24, 1, 1, 1)
    peak_gib = None
    if device.type == "cuda":
        # 采样阶段 cudaMallocAsync 池的保留残留会占着物理显存不还——进网络
        # 前先软清一次（不卸载任何模型），峰值统计也从零起算。
        comfy.model_management.soft_empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    parts = []
    with torch.inference_mode():
        # ── dense 前向（按窗分批）── 相位校正栅格（移植自 T8 dense_transfer_exp）：
        # 只替换编码器→解码器的空间映射，权重/层与嵌入约定与原实现一致。
        # 每窗独立"搬运→归一化→前向→反归一化→回 intermediate_device"，显存
        # 峰值 O(单窗) 而非 O(全片)——整片前向时单条 GroupNorm 激活就有
        # ~2.9GiB（13 镜 @1.5×），16GB 卡必炸（2026-11-10 9071.log 实录）。
        # intermediate_device 默认 CPU；--gpu-only 时驻留 GPU，峰值退回 O(全片)。
        for order, (first, last) in enumerate(spans):
            # 上下文头：窗首之前至多 _CONTEXT_FRAMES 帧真实内容一并喂入，输出侧
            # 裁掉——只补"过去"方向：下一窗的头能看到上一窗的真实尾部，各窗自己
            # 的尾帧仍无"未来"上下文（零填充，全片末帧本来就如此）。理论时序
            # 感受野单边 74 token，8 帧只覆盖近端（减轻而非消除边界效应）；
            # 随机初始化权重实测全局余弦 ~0.99、近缝逐帧余弦 ≥0.8，真实检查点
            # 漂移更小（tests/test_learned_upscale.py）。
            context = min(_CONTEXT_FRAMES, first)
            normalized = (video[:, :, first - context:last].to(device=device, dtype=dtype) - mean) / std
            chunk = _dense_network_forward(
                entry.model, normalized, effective_scale, tgt_latent_h, tgt_latent_w,
            ) * std + mean
            if context:
                chunk = chunk[:, :, context:]
            if not torch.isfinite(chunk).all():
                raise RuntimeError(
                    f"学习型放大器产生了 NaN/Inf 输出（窗 {order + 1}/{len(spans)}）"
                )
            parts.append(chunk.to(
                device=comfy.model_management.intermediate_device(), dtype=video.dtype,
            ))
            del normalized, chunk
    # 无条件 cat（单窗也拷一次）：块内 .to() 产出的是 inference tensor，cat 在
    # 块外执行才让 output_video 回归普通张量——与旧实现的输出契约一致，下游
    # （本包 refine/AVDecode 之外还有任意第三方节点）可能对它做就地写。
    output_video = torch.cat(parts, dim=2)
    del parts, mean, std

    out = dict(av_latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((output_video, audio))
    # mask 与前向共用同一批窗逐段重采样后 cat：nearest-exact 插值逐帧独立，
    # 拼装结果与整片一次逐位一致，只是峰值与大前向同步分批。非嵌套或畸形
    # （非 2-part）的 legacy mask 无法拆出音频半边——按 LKH3AVUpscale 的既有
    # 约定丢弃并留 debug 日志，绝不带着过期空间尺寸放行。
    masks = av_latent.get("noise_mask")
    mask_parts = tuple(masks.unbind()) if getattr(masks, "is_nested", False) else ()
    if len(mask_parts) == 2:
        video_mask, audio_mask = mask_parts
        resized_mask = torch.cat(
            [_upscale_video_chunk(
                video_mask[:, :, first:last], tgt_latent_w, tgt_latent_h, "nearest-exact",
            ) for first, last in spans],
            dim=2,
        )
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((resized_mask, audio_mask))
    elif masks is not None:
        # dict(av_latent) 已把旧 mask 带进 out——必须显式移除，否则下游会拿到
        # 空间尺寸不再匹配的过期 mask（比丢弃更糟）。
        out.pop("noise_mask", None)
        logging.debug(
            "LKH3AVLearnedUpscale dropped a non-nested/malformed noise_mask (%s); "
            "expected the 2-part nested AV mask from the long video sampler.",
            type(masks).__name__,
        )

    # 峰值/耗时口径覆盖前向 + 输出 cat + mask 拼装（mask 阶段也有一次全尺寸
    # cat）；release（模型回 CPU / 清缓存）不计入。
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1 << 30)
    elapsed = time.perf_counter() - started

    if release_policy != "keep_loaded":
        entry.model.to(device="cpu")
        if release_policy == "clear_after":
            _CACHE.clear()
            entry = None  # 松开局部强引用，下面的 gc.collect() 才能真正回收 CPU 权重
            gc.collect()
            comfy.model_management.soft_empty_cache()

    layout["video_hw"] = [tgt_latent_h, tgt_latent_w]
    out["lk_layout"] = layout
    report = {
        "schema_version": 1, "node": "LKH3AVLearnedUpscale", "status": "ok",
        "mode": "dense", "scale_by": scale, "effective_scale": round(effective_scale, 4),
        "source_latent": [src_h, src_w], "target_latent": [tgt_latent_h, tgt_latent_w],
        "target_px": [tgt_w, tgt_h], "precision": precision,
        "chunks": {
            "mode": chunk_mode, "effective_mode": effective_mode,
            "fallback": effective_mode != chunk_mode, "count": len(spans),
            "frames": [last - first for first, last in spans],
        },
    }
    label = _CHUNK_MODE_LABELS.get(effective_mode, effective_mode)
    if effective_mode != chunk_mode:
        label += f"（请求{_CHUNK_MODE_LABELS.get(chunk_mode, chunk_mode)}，已回退）"
    peak_note = ""
    if peak_gib is not None:
        peak_note = f" 峰值显存={peak_gib:.2f}GiB"
    logging.info(
        "[LKH3] 学习型放大(dense): (%d,%d)×%.3f→(%d,%d) 嵌入σ=%.3f | 目标画布 %dx%d"
        " | 分批=%s %d 窗%s 耗时=%.1fs",
        src_w, src_h, scale, tgt_latent_w, tgt_latent_h, effective_scale, tgt_w, tgt_h,
        label, len(spans), peak_note, elapsed,
    )
    return out, tgt_w, tgt_h, json.dumps(report, ensure_ascii=False, sort_keys=True)


class MiniMaxH3AVLearnedUpscaleGH(io.ComfyNode):
    """学习型放大（锚点推理 + 换算到目标画布）——LK 独立实现。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LKH3AVLearnedUpscale",
            display_name="MiniMax-H3 AV学习型放大（LK）",
            category="LK/MiniMax H3 Integration",
            description=(
                "Learned 3D latent resizer driven through the DENSE continuous-coordinate "
                "path (phase-corrected grid_sample transport) at the REAL geometric scale. "
                    "目标 = 一跑 latent × scale_by — the aspect ratio is closely preserved by construction "
                    "(even-token snap ⇒ /32-aligned canvas).  Continuous-scale conditioning "
                    "avoids the concentric-ring artifacts of naive phase-split interpolation "
                    "(reviewed at 1.2x); the refine pass stays at the familiar working canvas. "
                "The forward is temporally chunked per shot by default (per-window "
                "VRAM peak, long films never OOM on shot count); chunk_mode=whole "
                "keeps the legacy single-pass behavior for A/B."
            ),
            inputs=[
                io.Latent.Input(
                    "av_latent",
                    tooltip="一跑输出的整条嵌套 AV 潜空间（含 lk_layout）。非嵌套或畸形"
                            "（非 2-part）的 legacy noise_mask 会被丢弃（debug 日志），不做放大。",
                ),
                io.Combo.Input("model_name", options=_upscale_model_options()),
                io.Float.Input(
                    "scale_by", default=1.5, min=1.0, max=4.0, step=0.05,
                    tooltip="目标 = 一跑 latent × 此倍数（尺寸自动取偶 → 画布自动 /32，画幅比例基本保持）。H05 已审倍率 = 1.2；连续倍率均可。",
                ),
                io.Combo.Input("precision", options=list(PRECISIONS), default="fp16"),
                io.Combo.Input("release_policy", options=list(RELEASE_POLICIES), default="offload_after",
                               tooltip="offload_after = 推理后权重回 CPU（保留缓存）；clear_after = 连缓存一起清；keep_loaded = 常驻显存。"),
                io.Combo.Input("chunk_mode", options=list(CHUNK_MODES), default="segments",
                               tooltip="前向分批：segments = 按 lk_layout 分镜边界逐镜前向（默认，显存峰值 O(单镜)，长片不 OOM，GroupNorm 统计每镜一份更贴训练分布）；fixed = 固定窗口；whole = 整片一次（旧行为，供短片 A/B 与逐位复现）。"),
                io.Int.Input("chunk_window", default=DEFAULT_CHUNK_WINDOW, min=16, max=512, step=8,
                             tooltip="固定窗口模式每窗的潜空间帧数（96 窗 ≈ 324 实帧，仍在检查点训练 clip 长度内）；仅 fixed 模式生效，布局漂移回退时也用它。"),
            ],
            outputs=[
                io.Latent.Output("av_latent", display_name="AV 潜空间",
                                 tooltip="放大并换算后的嵌套 AV 潜空间（含更新后的 lk_layout）— 接二跑精修的 av_latent。"),
                io.Int.Output("width", display_name="宽度"),
                io.Int.Output("height", display_name="高度"),
                io.String.Output("report_json", display_name="报告"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        av_latent = kwargs.pop("av_latent")
        return io.NodeOutput(*learned_upscale_h3_av_latent(av_latent=av_latent, **kwargs))


NODES = [MiniMaxH3AVLearnedUpscaleGH]
