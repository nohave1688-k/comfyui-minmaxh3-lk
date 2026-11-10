"""学习型放大 分批前向 tests: chunk plan validity + chunked-vs-whole equivalence.

Run with the ComfyUI environment's python (no pytest needed):

    <aki>/python/python.exe tests/test_learned_upscale.py

Pinned claims:

1. ``_forward_chunk_plan``: ``segments`` spans mirror ``lk_layout`` exactly
   (contiguous, full coverage); ANY drift — coverage shortfall, missing
   segment, first offset nonzero, mid-span gap, structural garbage (missing
   keys / non-dict segments) — falls back to fixed windows AND records the
   fact (``effective_mode="fixed"``); window sizes clamp to >= 8; unknown
   modes fall back to fixed; ``total_t <= 0`` raises ``ValueError``.
2. ``fixed`` mode with window >= T is BIT-IDENTICAL to ``whole`` — the chunk
   plumbing adds nothing when a single window covers the film.  The
   layout-drift fallback result is bit-identical to the equivalent ``fixed``
   run.  Report ``chunks`` records requested mode, effective mode and the
   fallback flag honestly.
3. Multi-window forwards keep the output STRUCTURE and re-normalize the
   statistics.  This gap is NOT zero by design: GroupNorm statistics become
   per-window (the checkpoint is trained on single <=362-frame clips, so
   per-shot stats are closer to the training distribution than a 1000+-frame
   film's).  Measured on random-init weights at realistic window sizes
   (T=48, 2x24, context heads on): global cosine 0.989, per-frame-pooled
   cosine min 0.81, mean|drift| ~1.3% of scale, max|drift| ~0.66x scale (a
   tail of re-normalized channels).  The trained checkpoint's FiLM/residual
   magnitudes are smaller, so its drift is lower.
4. The noise-mask video half is resized per window with nearest-exact (a
   frame-local op) to the LATENT canvas and re-concatenated BIT-IDENTICALLY
   to a whole-tensor resize.
5. The audio halves pass through untouched (bit-exact): the samples audio
   half is asserted inside ``_run`` at EVERY forward (every chunk count is
   covered); the noise-mask audio half is asserted in the mask test.
6. Hostile inputs degrade, never crash: inf/structurally-broken segments and
   non-dict ``lk_layout`` fall back to fixed windows (entry level too, with
   ``fallback=True`` recorded); a 1-part nested mask is dropped like a
   non-nested one.
7. Non-square canvases keep their axes: fixtures use square latents, where an
   h/w swap would be invisible — a non-square pin feeds single-axis ramps and
   asserts the output varies along the fed axis only, plus exact target_px.
8. scale_by NaN falls back to 1.5 with a warning (never a silent clamp-to-1.0
   noop); scale_by=1.0 on an even canvas takes the noop path without loading
   the model; NaN-poisoned weights raise the per-window NaN/Inf RuntimeError.
"""
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_COMFY = os.path.abspath(os.path.join(_REPO, "..", ".."))
for path in (_COMFY, _REPO):
    if path not in sys.path:
        sys.path.insert(0, path)

import comfy.model_management  # noqa: E402
import comfy.nested_tensor  # noqa: E402

import gh.learned_upscale as lu  # noqa: E402
from gh.latent_upscale import _upscale_video_chunk  # noqa: E402


# ------------------------------------------------------------- chunk plan ----

def test_chunk_plan_whole_and_fixed():
    assert lu._forward_chunk_plan(12, None, "whole", 96) == ([(0, 12)], "whole")
    assert lu._forward_chunk_plan(12, None, "fixed", 8) == ([(0, 8), (8, 12)], "fixed")
    assert lu._forward_chunk_plan(24, None, "fixed", 8) == ([(0, 8), (8, 16), (16, 24)], "fixed")
    assert lu._forward_chunk_plan(10, None, "fixed", 10) == ([(0, 10)], "fixed")
    # 窗口下限钳到 8：防手滑/手改序列化值把每窗退化成逐帧。
    assert lu._forward_chunk_plan(12, None, "fixed", 3) == ([(0, 8), (8, 12)], "fixed")
    # 未知模式回退固定窗口（combo 之外的手改输入也安全），钳位同样生效。
    assert lu._forward_chunk_plan(12, None, "bogus", 6) == ([(0, 8), (8, 12)], "fixed")
    # T<=0：两条计划路径都无意义，直接拒绝并给出可读报错。
    for bad_t in (0, -4):
        try:
            lu._forward_chunk_plan(bad_t, None, "whole", 96)
            raise AssertionError(f"T={bad_t} should have raised ValueError")
        except ValueError:
            pass
    print("  ok  chunk plan: whole/fixed spans + window clamp + unknown mode + T<=0 guard")


def _layout(spans, video_hw=(4, 4)):
    segments = []
    offset = 0
    for tokens in spans:
        segments.append({
            "segment": len(segments) + 1,
            "video_offset": offset,
            "video_tokens": tokens,
            "audio_offset": offset * 8,
            "audio_steps": tokens * 8,
            "frames": tokens * 4,
        })
        offset += tokens
    return {
        "version": 1, "pass": "first", "latent_continue": True,
        "range": {"start": 1, "end": len(segments)},
        "video_hw": list(video_hw), "segments": segments,
    }


def test_chunk_plan_segments():
    assert lu._forward_chunk_plan(12, _layout([5, 7]), "segments", 96) == (
        [(0, 5), (5, 12)], "segments")
    assert lu._forward_chunk_plan(107, _layout([107]), "segments", 96) == (
        [(0, 107)], "segments")
    # 总覆盖不等于 T：回退固定窗口（窗口同样过 8 下限钳位）。
    assert lu._forward_chunk_plan(12, _layout([5, 5]), "segments", 6) == (
        [(0, 8), (8, 12)], "fixed")
    # 覆盖有缺（少一段）：回退。
    assert lu._forward_chunk_plan(12, _layout([4, 4]), "segments", 6) == (
        [(0, 8), (8, 12)], "fixed")
    # 中段断裂（首尾都对、中间 gap）：回退。
    midgap = _layout([4, 4, 4])
    midgap["segments"][1]["video_offset"] = 5  # spans (0,4),(5,9),(9,12)
    assert lu._forward_chunk_plan(12, midgap, "segments", 6) == (
        [(0, 8), (8, 12)], "fixed")
    # 首偏移非零：回退。
    drifted = _layout([5, 7])
    drifted["segments"][0]["video_offset"] = 1
    assert lu._forward_chunk_plan(12, drifted, "segments", 6) == (
        [(0, 8), (8, 12)], "fixed")
    # 结构异常（元素缺键）——按声明的意图回退而不是抛异常。
    broken = {"segments": [{"video_offset": 0}]}
    assert lu._forward_chunk_plan(12, broken, "segments", 6) == (
        [(0, 8), (8, 12)], "fixed")
    # layout 缺失（非本包 latent）：回退。
    assert lu._forward_chunk_plan(12, None, "segments", 6) == (
        [(0, 8), (8, 12)], "fixed")
    print("  ok  chunk plan: segments spans exact; shortfall/gap/offset/"
          "structural/missing layouts all fall back with effective_mode=fixed")


# ---------------------------------------------------------- forward passes ----

_TOTAL_T = 12
_MODEL = None


class _FakeEntry:
    def __init__(self, model):
        self.model = model


def _shared_model(seed=0):
    global _MODEL
    if _MODEL is None:
        # 默认 kaiming 初始化 + 固定种子：数值条件良好且可复现（randn 灌权会
        # 在 24 层残差块里指数爆炸，不适合做漂移基准）。
        torch.manual_seed(seed)
        model = lu.MiniMaxH3LearnedResizer3D()
        model.eval()
        _MODEL = model
    return _MODEL


def _fake_av_latent(total_t, h=4, w=4, seed=1, layout=None, with_mask=False):
    generator = torch.Generator().manual_seed(seed)
    video = torch.randn(1, 24, total_t, h, w, generator=generator)
    audio = torch.randn(1, 32, 2, total_t * 8, generator=generator)
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "gh_h3_audio_mode": "reference_only",
        "lk_layout": layout,
    }
    if with_mask:
        mask_video = torch.rand(1, 24, total_t, h, w, generator=generator).round()
        mask_audio = torch.rand(1, 32, 2, total_t * 8, generator=generator).round()
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_video, mask_audio))
    return latent


def _run(av_latent, chunk_mode, chunk_window, scale_by=1.5):
    input_video, input_audio = av_latent["samples"].unbind()
    out, width, height, report = lu.learned_upscale_h3_av_latent(
        av_latent=av_latent, model_name="fake.safetensors", scale_by=scale_by,
        precision="fp32", release_policy="offload_after",
        chunk_mode=chunk_mode, chunk_window=chunk_window,
    )
    video_out, audio_out = out["samples"].unbind()
    # audio 半边在任何窗数下都必须逐位原样透传（节点从不切音频）——每个
    # 前向自动覆盖，多窗路径不再有漏网。
    assert torch.equal(audio_out, input_audio), "audio passthrough changed"
    assert video_out.shape[2] == input_video.shape[2], "video T changed"
    return video_out, audio_out, out, int(width), int(height), json.loads(report)


def test_forward_equivalence():
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    lu._load_model = lambda name, precision: _FakeEntry(_shared_model())
    layout = _layout([5, 7])

    whole_video, _, _, w, h, whole_report = _run(
        _fake_av_latent(_TOTAL_T, layout=layout), "whole", 96,
    )
    assert (w, h) == (96, 96), f"unexpected pixel canvas {w}x{h}"  # 6x6 latent x16
    assert whole_video.shape == (1, 24, _TOTAL_T, 6, 6), whole_video.shape

    # 单窗覆盖全片：与整片逐位相等 —— 分批管道在单窗时零引入。
    single_video, _, _, _, _, single_report = _run(
        _fake_av_latent(_TOTAL_T, layout=layout), "fixed", _TOTAL_T,
    )
    assert torch.equal(whole_video, single_video), "single-window fixed diverged from whole"
    assert single_report["chunks"] == {
        "mode": "fixed", "effective_mode": "fixed", "fallback": False,
        "count": 1, "frames": [_TOTAL_T],
    }
    assert whole_report["chunks"] == {
        "mode": "whole", "effective_mode": "whole", "fallback": False,
        "count": 1, "frames": [_TOTAL_T],
    }
    print("  ok  fixed(window>=T) is bit-identical to whole; report records "
          "requested/effective mode + fallback=False")

    # 病理小窗（T=12、窗 8）：不按数值断言——8~12 帧的窗几乎每帧都贴零填充
    # 边界，时序卷积栈的边界污染主导输出（这正是实现里 _CONTEXT_FRAMES 上下文
    # 头存在的原因）。仅验证结构：形状、窗数、frames 内容、输出有限。
    # （window=5 会被 8 下限钳位 → (0,8),(8,12) 两窗；segments 按 [5,7] 布局两窗。）
    for mode, window, expected_frames in (("fixed", 5, [8, 4]), ("segments", 96, [5, 7])):
        chunk_video, _, _, _, _, report = _run(
            _fake_av_latent(_TOTAL_T, layout=layout), mode, window,
        )
        assert chunk_video.shape == whole_video.shape
        assert torch.isfinite(chunk_video).all()
        drift = (chunk_video - whole_video).abs().max().item()
        scale = whole_video.abs().max().item()
        print(f"  ok  {mode}(window={window}) vs whole: max|drift|={drift:.5f} "
              f"(scale={scale:.3f}, {len(expected_frames)} tiny windows, structural-only)")
        assert report["chunks"]["count"] == len(expected_frames), report["chunks"]
        assert report["chunks"]["frames"] == expected_frames, report["chunks"]
        assert report["chunks"]["fallback"] is False, report["chunks"]

    # 写实尺寸（T=48、每窗 24 ≈ 半个训练窗）：结构保持、统计重排。断言四个
    # 有意义的量（随机初始化权重上的实测形态，2026-11-10）：
    #   全局余弦 0.989、逐帧全通道余弦最小 ~0.81（近缝帧）、
    #   mean|drift| ~1.3% scale、max|drift| 是被重归一化通道的尾部（<1×scale）。
    # 真实检查点的 FiLM/残差分支幅度经过训练，漂移只会更小。
    big_layout = _layout([24, 24])
    big_video, _, _, _, _, _ = _run(
        _fake_av_latent(48, layout=big_layout, seed=2), "whole", 96,
    )
    assert big_video.shape == (1, 24, 48, 6, 6), big_video.shape
    big_scale = big_video.abs().max().item()
    chunked = {}
    for mode, window in (("fixed", 24), ("segments", 96)):
        video, _, _, _, _, report = _run(
            _fake_av_latent(48, layout=big_layout, seed=2), mode, window,
        )
        assert report["chunks"]["count"] == 2, report["chunks"]
        assert report["chunks"]["frames"] == [24, 24], report["chunks"]
        chunked[mode] = video
        diff = (video - big_video).abs()
        global_cos = torch.nn.functional.cosine_similarity(
            big_video.flatten(1), video.flatten(1), dim=1,
        ).item()
        # 真逐帧：每个 T 帧拼全通道后的向量余弦（比全局量严得多）。
        frame_a = big_video[0].permute(1, 0, 2, 3).flatten(1)
        frame_b = video[0].permute(1, 0, 2, 3).flatten(1)
        frame_cos = torch.nn.functional.cosine_similarity(frame_a, frame_b, dim=1)
        mean_rel = diff.mean().item() / big_scale
        assert global_cos >= 0.95, f"{mode} global cosine {global_cos:.4f}"
        assert frame_cos.min().item() >= 0.75, (
            f"{mode} per-frame cosine {frame_cos.min().item():.4f}"
        )
        assert mean_rel < 0.05, f"{mode} mean|drift| {mean_rel:.4f} rel scale"
        assert diff.max().item() < big_scale, f"{mode} max|drift| beyond output scale"
        print(f"  ok  {mode}(T=48, 2 windows, ctx-head) vs whole: "
              f"global_cos={global_cos:.4f} frame_cos>={frame_cos.min().item():.4f} "
              f"mean|drift|={mean_rel:.4f}xscale max|drift|={diff.max().item():.3f} "
              f"(scale={big_scale:.3f})")
    # segments[24,24] 与 fixed(24) 的保留跨度完全相同 → 结果逐位相等。
    assert torch.equal(chunked["fixed"], chunked["segments"]), (
        "identical spans must produce identical results"
    )

    # 布局漂移回退：segments(坏布局) 结果与 fixed(等参) 逐位相等，且 report
    # 如实记录"请求 segments、实际 fixed、fallback=True"。
    drifted = _layout([5, 5])
    bad_video, _, _, _, _, bad_report = _run(
        _fake_av_latent(_TOTAL_T, layout=drifted), "segments", 5,
    )
    ref_video, _, _, _, _, ref_report = _run(
        _fake_av_latent(_TOTAL_T, layout=drifted), "fixed", 5,
    )
    assert torch.equal(bad_video, ref_video), "layout-drift fallback diverged from fixed"
    assert bad_report["chunks"] == {
        "mode": "segments", "effective_mode": "fixed", "fallback": True,
        "count": 2, "frames": [8, 4],
    }
    print("  ok  layout-drift fallback matches fixed-window result bit-for-bit; "
          "report records fallback=True")


def test_noise_mask_per_window_resize():
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    lu._load_model = lambda name, precision: _FakeEntry(_shared_model())
    layout = _layout([5, 7])
    _, _, out, _, _, report = _run(
        _fake_av_latent(_TOTAL_T, layout=layout, with_mask=True), "segments", 96,
    )
    mask_video, mask_audio = out["noise_mask"].unbind()
    assert mask_video.shape == (1, 24, _TOTAL_T, 6, 6), mask_video.shape
    # nearest-exact 逐帧独立：逐窗拼装 == 整片一次（逐位）。mask 骑在 latent 上，
    # 缩放目标是 latent 画布（report["target_latent"]），不是像素画布。
    source = _fake_av_latent(_TOTAL_T, layout=layout, with_mask=True)
    whole_mask, input_mask_audio = source["noise_mask"].unbind()
    tgt_latent_h, tgt_latent_w = report["target_latent"]
    expected = _upscale_video_chunk(whole_mask, tgt_latent_w, tgt_latent_h, "nearest-exact")
    assert torch.equal(mask_video, expected), "per-window mask resize diverged from whole"
    # 音频半边的 mask 内容也必须逐位原样（samples 半边已由 _run 全程钉住）。
    assert torch.equal(mask_audio, input_mask_audio), "mask audio half changed"
    assert mask_audio.shape == (1, 32, 2, _TOTAL_T * 8), mask_audio.shape
    print("  ok  noise-mask: video half per-window cat bit-identical, "
          "audio half bit-exact passthrough")


def test_non_nested_mask_degrades():
    # legacy 纯张量 mask（如 SetLatentNoiseMask 直连）：无法拆音频半边——
    # 按 LKH3AVUpscale 的既有约定丢弃（debug 日志）而不是晦涩崩溃。
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    lu._load_model = lambda name, precision: _FakeEntry(_shared_model())
    latent = _fake_av_latent(_TOTAL_T, layout=_layout([5, 7]))
    plain_mask = torch.rand(1, 1, _TOTAL_T, 4, 4).round()
    latent["noise_mask"] = plain_mask  # 非 NestedTensor
    _, _, out, _, _, _ = _run(latent, "segments", 96)
    assert "noise_mask" not in out, "non-nested mask should be dropped, not crashed on"
    # 1-part 嵌套 mask（畸形）：同样丢弃，不崩。
    latent2 = _fake_av_latent(_TOTAL_T, layout=_layout([5, 7]))
    latent2["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.rand(1, 1, _TOTAL_T, 4, 4).round(),),
    )
    _, _, out2, _, _, _ = _run(latent2, "segments", 96)
    assert "noise_mask" not in out2, "malformed 1-part nested mask should be dropped"
    print("  ok  non-nested / malformed (1-part) masks dropped gracefully (no crash)")


def test_hostile_layout_inputs_degrade():
    # 计划层：inf offset（int(inf) 抛 OverflowError）、非 dict layout——都回退 fixed。
    inf_layout = {"segments": [{"video_offset": float("inf"), "video_tokens": 12}]}
    assert lu._forward_chunk_plan(12, inf_layout, "segments", 6) == (
        [(0, 8), (8, 12)], "fixed")
    for bad_layout in (5, "x", ["not", "a", "dict"], {"segments": "nope"},
                       {"segments": [None, 3, "bad"]}):
        assert lu._forward_chunk_plan(12, bad_layout, "segments", 6) == (
            [(0, 8), (8, 12)], "fixed"), f"bad layout {bad_layout!r} should fall back"
    # 入口层：非 dict lk_layout 跑完整条链路不炸（回退 fixed + fallback=True）。
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    lu._load_model = lambda name, precision: _FakeEntry(_shared_model())
    latent = _fake_av_latent(_TOTAL_T)
    latent["lk_layout"] = 5  # 手改/外来 latent 的垃圾布局
    _, _, _, _, _, report = _run(latent, "segments", 6)
    assert report["chunks"] == {
        "mode": "segments", "effective_mode": "fixed", "fallback": True,
        "count": 2, "frames": [8, 4],
    }
    print("  ok  hostile layouts (inf/garbage/non-dict) fall back at plan and "
          "entry level with fallback=True")


def test_non_square_axis_order():
    # 非方形画幅钉证：全套夹具都是正方形，h/w 互换类回归在正方形下逐位不可见。
    # (a) 直接钉 dense 栅格的轴契约：[...,0]=x 只随目标 W 变、[...,1]=y 只随
    #     目标 H 变。数值是 patch 单位的仿射映射（等面积退化时才退化为
    #     align_corners 的 linspace），所以这里钉排他/单调/镜像三性质，
    #     不复制公式。
    grid = lu._dense_patch_grid((4, 6), (6, 10), device=torch.device("cpu"))
    assert grid.shape == (1, 6, 10, 2), grid.shape
    x_axis, y_axis = grid[0, :, :, 0], grid[0, :, :, 1]
    # 轴排他：x 只随目标 W 变（沿 H 恒定、严格递增），y 只随目标 H 变。
    assert torch.equal(x_axis, x_axis[0:1].expand(6, 10)), (
        "grid x-axis must be constant along H")
    assert torch.equal(y_axis, y_axis[:, 0:1].expand(6, 10).contiguous()), (
        "grid y-axis must be constant along W")
    assert (x_axis[0, 1:] > x_axis[0, :-1]).all() and (y_axis[1:, 0] > y_axis[:-1, 0]).all(), (
        "grid axes must be strictly monotonic")
    # 注意：dense 目标场覆盖更大画布，轴值有意越出 align_corners 的 [-1,1]
    # （由 grid_sample 的 padding_mode="border" 截边，_resize_dense_field 注释），
    # 所以这里不设值域断言。
    # 镜像属性：交换 H/W 的调用给出 x/y 互换的同一栅格——轴序互换回归的直接反证
    # （两次调用的 (source_area, target_area) 相同，映射数值逐位一致）。
    grid_swapped = lu._dense_patch_grid((6, 4), (10, 6), device=torch.device("cpu"))
    assert grid_swapped.shape == (1, 10, 6, 2), grid_swapped.shape
    assert torch.equal(grid_swapped[0, :, :, 0].T.contiguous(), y_axis), (
        "swapped-call x must equal original y (axis mirror)")
    assert torch.equal(grid_swapped[0, :, :, 1].T.contiguous(), x_axis), (
        "swapped-call y must equal original x (axis mirror)")
    # (b) 端到端形状/target_px/mask：4x6 @1.5 → 6x10 latent（160x96 像素）。
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    lu._load_model = lambda name, precision: _FakeEntry(_shared_model())
    layout = _layout([4], video_hw=(4, 6))
    _, _, out, w, h, report = _run(
        _fake_av_latent(4, h=4, w=6, layout=layout, with_mask=True), "whole", 96,
    )
    video_out = out["samples"].unbind()[0]
    mask_video = out["noise_mask"].unbind()[0]
    assert video_out.shape == (1, 24, 4, 6, 10), video_out.shape
    assert mask_video.shape == (1, 24, 4, 6, 10), mask_video.shape
    assert (w, h) == (160, 96) and report["target_px"] == [160, 96], report["target_px"]
    print("  ok  non-square 4x6 @1.5 -> 6x10 (160x96 px): grid x/y axis contract "
          "pinned exactly; end-to-end video/mask shapes + target_px match")


def test_noop_and_nan_scale_guard():
    # noop 路径端到端：偶数画幅 scale 1.0 → 不加载模型、video 原样、layout 补 video_hw。
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    calls = []
    def _spy_load(name, precision):
        calls.append(name)
        return _FakeEntry(_shared_model())
    lu._load_model = _spy_load
    layout = _layout([5, 7])
    latent = _fake_av_latent(_TOTAL_T, layout=layout)
    input_video, _ = latent["samples"].unbind()
    video, _, out, w, h, report = _run(latent, "whole", 96, scale_by=1.0)
    assert report["status"] == "noop" and calls == [], report["status"]
    assert (w, h) == (64, 64), (w, h)  # 4x4 latent x16
    assert torch.equal(video, input_video), "noop must not touch samples"
    assert out["lk_layout"]["video_hw"] == [4, 4], out["lk_layout"]
    # NaN scale_by：不再静默钳成 1.0 变无声 noop——告警 + 回退默认 1.5。
    video2, _, _, _, _, report2 = _run(latent, "whole", 96, scale_by=float("nan"))
    assert report2["status"] == "ok" and calls, report2["status"]
    assert abs(report2["effective_scale"] - 1.5) < 1e-3, report2["effective_scale"]
    assert video2.shape == (1, 24, _TOTAL_T, 6, 6), video2.shape
    print("  ok  scale_by=1.0 noop (no model load, samples untouched, video_hw filled); "
          "NaN scale_by warns + falls back to 1.5")


def test_nan_output_guard():
    # NaN 权重 → 每窗 isfinite 守卫必须抛带窗序号的 RuntimeError，不得静默出坏帧。
    torch.manual_seed(11)
    poisoned = lu.MiniMaxH3LearnedResizer3D()
    with torch.no_grad():
        poisoned.conv_out.weight[0, 0, 0, 0, 0] = float("nan")
    poisoned.eval()
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    lu._load_model = lambda name, precision: _FakeEntry(poisoned)
    try:
        _run(_fake_av_latent(4, layout=_layout([4])), "whole", 96)
        raise AssertionError("NaN weights should have raised a RuntimeError")
    except RuntimeError as exc:
        assert "NaN/Inf" in str(exc) and "窗" in str(exc), str(exc)
    print("  ok  NaN-poisoned weights raise per-window NaN/Inf RuntimeError")


def main():
    print("chunk plan:")
    test_chunk_plan_whole_and_fixed()
    test_chunk_plan_segments()
    print("forward equivalence (random-weight model, CPU):")
    test_forward_equivalence()
    test_noise_mask_per_window_resize()
    test_non_nested_mask_degrades()
    test_hostile_layout_inputs_degrade()
    test_non_square_axis_order()
    test_noop_and_nan_scale_guard()
    test_nan_output_guard()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
