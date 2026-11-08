# ComfyUI MiniMax H3 LK (Long Video)

> MiniMax H3 分段多镜头音视频生成插件 —— ComfyUI-MiniMaxH3-lanxi 的重建版。
>
> T8 式两跑高清化：分段引导一采（T8 原语分解）→ 潜空间放大 → 二跑分段精修，
> 全程打包 AV 潜空间（视频+音频联合生成、联合解码）。

**English intro:** Segmented multi-shot **audio+video** generation for the MiniMax H3
hybrid Ref2VA model. One-stop film composer (whole-film prompt + per-shot LoRA/RefMod
panels) → segmented first pass → packed AV-latent upscale → per-shot second-pass
refine → joint audio-video decode. 7 nodes; reference-photo identity sheets
(YuNet face crops) for character consistency. Full docs below are in Chinese.

## 节点总览（7 个）

| 节点 | 显示名 | 角色 |
|---|---|---|
| `LKH3Composer` | MiniMaxH3 长视频编排（LK） | 一站式编排：全片提示词 + 模型选择 + 分镜收集 + 空白 AV 模板 |
| `LKH3Shot` | MiniMaxH3分镜（LK） | 单个镜头：时长、提示词、参考媒体、LoRA/RefMod 行（面板管理） |
| `LKH3SegmentGuider` | MiniMax-H3 分段引导（LK） | 一跑 GUIDER：持有分段循环（逐镜条件化 + LoRA + latent 续接），自带 SAMPLER/SIGMAS |
| `LKH3RefineSegments` | MiniMax-H3 二跑分段精修（LK） | 二跑：放大后的整条潜空间逐镜精修（HD 条件化重编码、精修调度、一跑音频原样保留） |
| `LKH3AVUpscale` | MiniMax-H3 AV潜空间放大（LK） | 打包 AV 潜空间的空间放大（只插值视频部分） |
| `LKH3ReferenceFocus` | MiniMax-H3 参考聚焦（LK） | 参考照片预处理：YuNet 人脸检测 → 身份特写表，强化角色一致性 |
| `LKH3AVDecode` | MiniMax-H3 AV解码（LK） | 打包 AV 潜空间 → 帧 + 音频（按消费点解码） |

分类统一在 `LK/MiniMax H3 Integration`；分段计划插座类型为 `LANXI_H3_SEGMENTS`
（刻意沿用旧名：计划结构与 lanxi 前身包的长视频采样器互通）。

## 标准工作流（一跑，T8 原语分解）

```
LKH3Shot ×1-8 ─shot─▶ LKH3Composer ─segments────────▶ LKH3SegmentGuider ─GUIDER┐
                             │           ├─空白AV模板─▶ SamplerCustomAdvanced ◀─SIGMAS┤
                             │           │                ▲noise（RandomNoise 原语） │
                             │           └─CLIP/VAE────▶ （按需连给引导/精修/解码）    ─SAMPLER┘
                             │
                             └─ SamplerCustomAdvanced 输出 LATENT ─▶ LKH3AVDecode ─▶ 图像 + 音频
```

- **LKH3Composer**：`film_prompt` 拼到每个镜头提示词前（主体定义/风格块/`<图片n>` 标签）；
  `aspect`+`megapixels`（默认 0.5MP）定全局画布；`audio_mode`（lock_source / reference_only /
  native）管音频来源；`latent_continue` 开关段间潜空间续接；`range_start/end` 选跑哪些镜头。
  输出的 **空白AV模板**（含 `lk_layout` 布局）直接接 SamplerCustomAdvanced 的 `latent_image`，
  不再需要独立的模板/断环节点。
- **LKH3Shot**：时长 2-30s（自动对齐 17n+5 帧网格）；参考媒体走原生连线插座
  （首/尾帧、参考图×6、参考视频×2、参考音频×3）；LoRA/RefMod 行由前端面板管理
  （序列化在 `shot_json`）。
- **LKH3SegmentGuider**：把 SamplerCustomAdvanced 拆出的原语组合成分段循环——
  逐镜条件化 + 逐镜 LoRA + 潜空间续接（上一镜尾部 7 latent token / 22 帧）。
  SAMPLER/SIGMAS 由本节点从同一份 steps/shifts 配方生成（单一事实来源，无独立契约节点）。
  每镜采样器按该镜自己的音频长度重建（整片契约采样器会错切分段张量，见节点描述）。

## 高清化工作流（二跑）

```
一跑 LATENT ─▶ LKH3AVUpscale（scale_by 2.0 → ~1080p）─▶ LKH3RefineSegments ─▶ LKH3AVDecode ─▶ 高清成片
                                                            ▲ segments/CLIP/VAE 从 LKH3Composer 连线
```

- **LKH3AVUpscale**：只对视频部分做空间插值（nearest-exact/bilinear/area/bicubic/lanczos，
  默认 bilinear、2.0× = 0.5MP 基础画布 → H3 训练上限 ~1920×1088）；音频、mask 与
  `lk_layout` 元数据原样透传。
- **LKH3RefineSegments**：逐镜切片精修，条件化按**放大后的画布**重建（参考图 HD 重编码，
  等价 T8 的 HIGH-conditioning）。精修调度 = base_steps(9) 基础调度从 coarse_steps(4) 截断的
  尾段（精修步数 = base − coarse），入口去噪 `refine_entry_sigma` 默认 0.30（T8 发布配方：
  simple 调度、denoise 0.30，二跑是轻度润色而非重生成）。噪声来自**全片共享的全局噪声场**
  按位置切片，邻镜重建相互关联。`carry_source`：anchor（固定质量锚点，推荐）/ chained
  （链式接缝）/ none（诊断）。`audio_policy="keep"` 时一跑音频逐位保留
  （音频与分辨率无关，二跑不重写）。

## 关键机制

- **条件化预构建**：一跑/二跑的所有镜头条件化在采样前一次性构建，文本编码器整轮
  只装载/卸载各一次——多镜头视频的采样阶段零 CLIP 搬运；`release_clip` 语义为
  "构建完成后、首镜采样前释放一次"（显存紧张仍应保持开启；16GB 卡两个 advanced
  开关都建议保持默认 true）
- **潜空间续接**：上一镜尾部 7 个 latent token（=22 帧）+ 36 音频步写入下一镜头部，
  由 denoise_mask 锁定；输出侧裁掉重复帧，全片逐帧连续
- **打包 AV 潜空间**：视频 [1,24,T,H/16,W/16] + 音频 [1,32,2,T_a] 的 NestedTensor
  （`lk_layout` 记录逐镜布局），是所有采样/放大/解码节点的通用交换格式

## 与旧 lanxi 包的关系

| | lanxi（旧） | 本包 |
|---|---|---|
| 节点 | 7 个已注册 + 16 个未注册 RefMod | **7 个**（编排/分镜/引导/精修/放大/聚焦/解码） |
| 一跑形态 | 适配器 + 一体化采样器 | T8 原语分解（GUIDER + SamplerCustomAdvanced） |
| 二跑形态 | 二采分块采样器 + Latent 上采样器 | `LKH3AVUpscale` + `LKH3RefineSegments` |
| 注册方式 | V1 双轨 + comfy_entrypoint | 纯 `NODE_CLASS_MAPPINGS`（V3 类原生执行） |
| 节点 ID | `LanxiH3Studio` 等 | `LKH3*`（可共存） |
| 计划插座 | `LANXI_H3_SEGMENTS` | 同名同构，**计划可互换** |
| RefMod 库 | `models/refmods/` | 同一路径，**两包共享** |
| API 路由 | `/lanxi/...` | `/lk/...` |

## 安装

### 方式一：ComfyUI Manager（推荐）

Manager → **Custom Nodes Manager** → **Install via Git URL**，粘贴：

```
https://github.com/nohave1688-k/comfyui-minmaxh3-lk
```

Manager 会克隆仓库并自动安装 `requirements.txt` 中的依赖。

### 方式二：手动 git clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nohave1688-k/comfyui-minmaxh3-lk.git
```

重启 ComfyUI。如需手动补依赖：

```bash
pip install -r ComfyUI/custom_nodes/comfyui-minmaxh3-lk/requirements.txt
```

### 依赖说明

- 硬依赖（`requirements.txt`，Manager 自动安装）：safetensors / numpy / Pillow /
  sentencepiece / transformers≥4.49 / accelerate / imageio-ffmpeg≥0.6.0（torch 由 ComfyUI 提供）
- **参考聚焦**节点需要 `opencv-python`（YuNet 人脸检测器）；RefMod 视频加载器可选装
  opencv 或 imageio 之一
- 需要 MiniMax H3（Ref2VA 混合）模型文件放置于 ComfyUI 常规模型目录

## 结构

```
gh/            引擎（编排/分镜、条件化、双时钟采样、预构建循环、精修、AV解码、参考聚焦、提示词优化）
refmod/        RefMod 参考库（面板底层，存储于 models/refmods/）
web/js/        统一前端面板（分镜 LoRA/RefMod 行、参考媒体、提示词优化器）
locales/       中英文翻译（nodeDefs.json 按节点/输入名组织）
models/        face_detection_yunet_2023mar.onnx（参考聚焦内置检测器，230KB）
```

## API 路由

`/lk/minimax_h3/loras`（LoRA 列表）、`/lk/refmods/library`（RefMod 库）、
`/lk/minimax-h3/prompt-optimizer/*`（提示词优化器配置/模型/状态）。

## 增量开发

- 分镜数据契约：`gh/nodes.py` 的 shot 计划构建与前端 `web/js/minimax_h3_unified.js`
  的 `shot_json` 序列化需同步
- 新增输入/输出记得同步 `locales/{zh,en}/nodeDefs.json`（按输入名与输出索引组织，
  缺条目时回退英文名）
- 采样语义改动先读 `gh/long_video.py` 的 `run_long_video` / `run_refine_segments`
  与 `gh/sampling.py` 的双时钟实现；二跑调度的 T8 对齐依据写在节点 tooltip 与
  `run_refine_segments` docstring 里

## 致谢 / Credits

- [OpenCV Zoo — YuNet](https://github.com/opencv/opencv_zoo)：
  `models/face_detection_yunet_2023mar.onnx` 人脸检测模型（Apache-2.0），
  供 `LKH3ReferenceFocus` 内置使用，随包分发以便开箱即用
- 前身包 ComfyUI-MiniMaxH3-lanxi 的分段计划契约（`LANXI_H3_SEGMENTS`）与
  RefMod 库路径在本包延续
- 两跑高清化精修配方对齐 T8 发布参数（simple 调度、入口去噪 0.30）

## 许可证 / License

[MIT](LICENSE)。内置的 YuNet 检测器模型遵循其原始 Apache-2.0 许可证
（[OpenCV Zoo](https://github.com/opencv/opencv_zoo)）。
