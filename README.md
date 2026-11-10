# ComfyUI MiniMax H3 LK (Long Video)

> MiniMax H3 分段多镜头音视频生成插件 —— ComfyUI-MiniMaxH3-lanxi 的重建版。
>
> T8 式两跑高清化：分段引导一采（T8 原语分解）→ 潜空间放大 → 二跑分段精修，
> 全程打包 AV 潜空间（视频+音频联合生成、联合解码）。

**English intro:** Segmented multi-shot **audio+video** generation for the MiniMax H3
hybrid Ref2VA model. One-stop film composer (whole-film prompt + per-shot LoRA
panels) → segmented first pass → packed AV-latent upscale → per-shot second-pass
refine → joint audio-video decode. 9 nodes. Full docs below are in Chinese.

## 节点总览（9 个）

| 节点 | 显示名 | 角色 |
|---|---|---|
| `LKH3Composer` | MiniMaxH3 长视频编排（LK） | 一站式编排：全片提示词 + 模型选择 + 分镜收集 + 空白 AV 模板 |
| `LKH3Shot` | MiniMaxH3分镜（LK） | 单个镜头：时长、提示词、参考媒体、LoRA 行（面板管理） |
| `LKH3ShotGroup` | MiniMaxH3 分镜组（LK） | 一组 ≤3 镜：分镜1（主）全套参考 socket，分镜2/3 面板编辑 + 默认继承主分镜参考；组链首尾相接解除 8 镜上限 |
| `LKH3SegmentGuider` | MiniMax-H3 分段引导（LK） | 一跑 GUIDER：持有分段循环（逐镜条件化 + LoRA + latent 续接），自带 SAMPLER/SIGMAS |
| `LKH3RefineSegments` | MiniMax-H3 二跑分段精修（LK） | 二跑：放大后的整条潜空间逐镜精修（HD 条件化重编码、精修调度、一跑音频原样保留） |
| `LKH3AVUpscale` | MiniMax-H3 AV潜空间放大（LK） | 打包 AV 潜空间的空间放大（只插值视频部分） |
| `LKH3AVLearnedUpscale` | MiniMax-H3 AV学习型放大（LK） | 用 H3 放大检查点做 T8 dense 前向（相位校正栅格，连续倍率可用）；前向默认按分镜边界分批，显存峰值 O(单镜)，长片不随分镜数上涨（`前向分批=整片一次` 可回旧行为） |
| `LKH3AVDecode` | MiniMax-H3 AV解码（LK） | 打包 AV 潜空间 → 帧 + 音频（按消费点解码） |
| `LKH3AVDiskDecode` | MiniMax-H3 AV落盘解码（LK） | 落盘出片：按模型自身时间分块协议逐块解码（内存有界） + 分段 H.264 流水线编码 → stream-copy 拼接整片 MP4 |

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
  （首/尾帧、参考图×6、参考视频×2、参考音频×3）；LoRA 行由前端面板管理
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
  默认 bilinear、2.0× = 0.5MP 基础画布 → H3 训练上限 ~1920×1088）；音频与
  `lk_layout` 元数据原样透传，mask 视频半边随目标画布重采样。长片（分镜多、
  放大后显存吃紧）可改用 **LKH3AVLearnedUpscale**：H3 放大检查点的 dense 前向，
  时间维按分镜分批，显存峰值 O(单镜)；两者的 latent 输出可直接互换接入二跑
  （学习型放大额外多 width/height/report_json 三个信息输出口；`lk_layout.video_hw`
  由学习型放大维护，插值节点原样透传）。二跑解码阶段长片建议改接
  **LKH3AVDiskDecode**（逐段流式出片，内存有界）。
- **LKH3RefineSegments**：逐镜切片精修，条件化按**放大后的画布**重建（参考图 HD 重编码，
  等价 T8 的 HIGH-conditioning）。精修调度为**入口 σ 线性降到 0**：`refine_entry_sigma`
  就是精确的入口去噪量（默认 0.30，落在 pika 官方网格低噪区；1.0 = 旧截断调度全量重生成），
  精修步数 = base − coarse。噪声来自**全片共享的全局噪声场**按位置切片，邻镜重建相互关联。
  `carry_source`：anchor（固定质量锚点，推荐）/ chained（链式接缝）/ none（诊断）。
  `audio_policy="keep"` 时一跑音频逐位保留（音频与分辨率无关，二跑不重写）。

## 逐镜缓存（项目束，全自动）

无需任何配置：编排节点自动在 `output/LK_H3_projects/` 下按片建缓存束
（「项目目录」留空即用默认位置，也可填自定义路径）——改哪镜只重跑哪镜：

```
{缓存束}/
├─ manifest.json       逐段缓存键（机器读）
└─ segNNN_first.latent / segNNN_second.latent   分段缓存
```

- **分段缓存**：改哪镜只重跑哪镜。缓存键递归包含上游内容 hash + 参考媒体
  内容 hash——模型/提示词/LoRA/参考/参数任一变化，对应段正确 miss，绝不
  静默用旧数据。`range` 从中间起跑时，**上一镜规格未变**才会载入其尾部
  latent 作续接（接缝与整片一致）；上一镜本身变了（含换画布/换参考导致
  的尺寸变化）则冷启动，绝不接错形状的尾巴。
- **分镜组自动兼容**：「分镜组」节点展平后走同一 plan 通道——逐镜 audio_mode、
  媒体内容 hash、缓存键全部自动生效，无需任何额外配置。
- **身份**：束的身份只含模型/画布/镜数——改某一镜的提示词不会换束；结构变更
  （换模型、改画布、增删镜头）会开新束。
- **全自动清理**：束内容全部可再生。全部束的磁盘占用超过 `LK_BUNDLE_CACHE_GB`
  （默认 16GB，设 0 关闭）按最旧优先**整束删除**；`LK_BUNDLE_CACHE_DAYS`
  可按最后活动时间清理（默认 0 = 关）。
- 开关：`LK_SEGMENT_CACHE=0` 完全关闭缓存（不建目录）。

## 关键机制

- **逐镜懒构建 + `release_clip` 默认关闭**：每镜条件化在采样前即时构建（历史上的
  CLIP 装载/释放节奏）。**不要开启 release_clip**——int8 量化的 32B 文本编码器卸载后
  热重载的量化状态与首次不一致，参考嵌入会退化：表现为**重启后首轮参考正常、之后每轮
  排队参考逐渐丢失**（实测复现）。开启仅省显存，代价是参考稳定性
- **提示词与参考图必须一致**：提示词里的外观/服装/配件描述要和当前接的参考图匹配
  （换参考图后需同步重新生成或改写提示词）——文本描述旧角色 + 图像是新角色 =
  条件化互相矛盾，直接表现为"不像参考、不遵循提示词"
- **参考图尺寸建议 `match`**：参考图按画布面积匹配缩放后再进入条件化（与 T8 官方
  配方一致）。`max` 会把大图原样送入，视觉 token 数暴涨，在新版 comfy 的视觉预算下
  可能被截断
- **节点每队列全新构建**：编排/引导/精修三个管线节点声明 `NOT_IDEMPOTENT`，
  条件化与状态不跨队列复用（杜绝跨运行的脏状态）
- **NaN 防御**：精修的 `context_noise`/`refine_entry_sigma` 等浮点输入若因旧画布
  错位收到 NaN/非法值，自动回退默认值并打警告（日志关键词"非有限数值输入"），
  不再污染掩码与调度
- **采样调度日志带阶段标签**：`采样调度(一采镜头N)` / `采样调度(二跑窗口N)` 各自
  打印实际使用的 σ 表——蒸馏 LoRA 只在自家 σ 轨迹上服从条件化，排障时先看这两行
- **LK_debug 诊断**：每次运行自动写入 `user/LK_debug/`——参考图编码前后的存证 PNG、
  潜空间统计（`refdbg.jsonl`）、参考潜空间回解码存证（`reflatent_*.png`）。参考不像时
  先看参考潜空间回解码：解码出来是本人 = 条件化无罪，问题在采样侧
- **锐度锚（长片防软化）**：分段引导的 advanced 开关。每镜采样后解码其末帧，作为
  下一镜的**附加参考图**——时间注意力按上一镜真实锐度匹配，阻断 latent 续接链的
  "逐镜向软漂移"（photocopy 效应，实测把劣化起点从 ~30s 推到 40s+）。与手动参考
  并存（占下一个空闲参考槽，满 9 张时本镜跳过并告警）；开启时自动切回逐镜懒构建
- **潜空间续接**：上一镜尾部 7 个 latent token（=22 帧）+ 36 音频步写入下一镜头部，
  由 denoise_mask 锁定；输出侧裁掉重复帧，全片逐帧连续
- **打包 AV 潜空间**：视频 [1,24,T,H/16,W/16] + 音频 [1,32,2,T_a] 的 NestedTensor
  （`lk_layout` 记录逐镜布局），是所有采样/放大/解码节点的通用交换格式

## 与旧 lanxi 包的关系

| | lanxi（旧） | 本包 |
|---|---|---|
| 节点 | 7 个已注册 + 16 个未注册 RefMod | **9 个**（编排/分镜/分镜组/引导/精修/放大×2/解码×2；RefMod 子系统已移除） |
| 一跑形态 | 适配器 + 一体化采样器 | T8 原语分解（GUIDER + SamplerCustomAdvanced） |
| 二跑形态 | 二采分块采样器 + Latent 上采样器 | `LKH3AVUpscale` + `LKH3RefineSegments` |
| 注册方式 | V1 双轨 + comfy_entrypoint | 纯 `NODE_CLASS_MAPPINGS`（V3 类原生执行） |
| 节点 ID | `LanxiH3Studio` 等 | `LKH3*`（可共存） |
| 计划插座 | `LANXI_H3_SEGMENTS` | 同名同构，**计划可互换** |
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
- 需要 MiniMax H3（Ref2VA 混合）模型文件放置于 ComfyUI 常规模型目录

## 升级与工作流兼容性

- 节点定义只来自安装的代码：**git 更新 → 重启 ComfyUI → 浏览器 Ctrl+F5**，三步完成升级，
  所有工作流的节点定义随之刷新；工作流文件只保存连线与控件数值
- v1.3.0 移除了 `LKH3ReferenceFocus`：打开含这个节点的旧工作流会提示节点缺失——
  参考照片直接接入分镜的参考插座即可。`LKH3AVDiskDecode` 在包内注册可用
  （长片流式出片推荐路径），无需改线
- v1.2.0 起二跑精修的 `放大模型`/`放大倍率`（内部学习型放大）已移除——放大统一走
  外接放大节点（当时为 `LKH3AVUpscale` 插值；v1.3.0 起另有 `LKH3AVLearnedUpscale`
  检查点 dense 前向可选）。
  **打开旧工作流若见 `续接头重加噪 = NaN` 或 `精修去噪强度 = 1.00`，
  说明该画布是旧槽位时代的存档**：重新从放大节点连线、或对照 V14 模板核对精修控件值
  （`context_noise 0.15` / `refine_entry_sigma 0.30`）。包内已有 NaN 防御——错位值会回退
  默认并打警告（日志关键词"非有限数值输入"），不会静默出坏图
- 控件纪律：**widget 只增不删不换序**；弃用的控件保留为 no-op 默认值

## 更新日志 / Changelog

**v1.3.0**

- 新增 `LKH3AVLearnedUpscale`：H3 放大检查点的 dense 前向（相位校正栅格，连续倍率可用），
  时间维默认**按分镜分批**（每窗至多 8 帧上下文头），显存峰值 O(单镜)，长片不再随分镜数 OOM；
  `前向分批=整片一次` 保留旧行为供 A/B 对照
- 项目束退化为**纯逐镜性能缓存**：目录自动生成（`output/LK_H3_projects/`，
  「项目目录」留空即用），束内容全部可再生；移除复现说明/参考媒体副本/
  「接回」入口（复现工作流放弃，接回按钮与弹窗已删）
- 缓存全自动清理：`LK_BUNDLE_CACHE_GB`（默认 16GB，0=关）超限最旧优先
  **整束删除**；`LK_BUNDLE_CACHE_DAYS` 按最后活动时间清理（默认 0 = 关）；
  `LK_SEGMENT_CACHE=0` 完全关闭缓存（不建目录）
- `LKH3ReferenceFocus`（参考聚焦）移除：参考照片直接接分镜的参考插座
- 前端：全局编号引擎修复（直接接编排的组误判「未接编排」）+ 分镜卡编号
  对齐后端展平口径（空主镜不占号）
- 诊断：参考潜空间回解码存证加形状守卫（不再每镜报 PIL 错误）

**v1.2.0**

- 新增 `LKH3AVDiskDecode`：逐段解码 + ffmpeg 增量编码，长片不再撞整片解码的内存墙
- 二跑精修移除内部学习型放大——放大统一走 `LKH3AVUpscale` 外接节点
- `release_clip` 默认改为 **false**：int8 量化 32B 文本编码器卸载后热重载会退化参考嵌入
  （首轮正常、后续排队参考丢失，实测复现）
- 一采/二跑改为**逐镜懒构建**（条件化构建节奏回到参考无退化的历史行为）
- 编排/引导/精修声明 `NOT_IDEMPOTENT`：管线状态不跨队列复用
- NaN 防御、逐镜懒构建、调度日志阶段标签（`一采镜头N`/`二跑窗口N`）、
  `user/LK_debug/` 参考编码前后存证与统计（`refdbg.jsonl`）

**v1.1.0 及更早**

- T8 式两跑高清化（一采分段引导 + `LKH3AVUpscale` + 二跑分段精修）、参考聚焦、
  打包 AV 潜空间契约、提示词优化器、中英文面板

## 结构

```
gh/            引擎（编排/分镜、条件化、双时钟采样、预构建循环、精修、AV解码、提示词优化）
web/js/        统一前端面板（分镜 LoRA 行、参考媒体、提示词优化器）
locales/       中英文翻译（nodeDefs.json 按节点/输入名组织）
```

## API 路由

`/lk/minimax_h3/loras`（LoRA 列表）、
`/lk/minimax-h3/prompt-optimizer/*`（提示词优化器配置/模型/状态）。

## 增量开发

- 分镜数据契约：`gh/nodes.py` 的 shot 计划构建与前端 `web/js/minimax_h3_unified_v3.js`
  的 `shot_json` 序列化需同步
- **widget 纪律：只增不删不换序**——工作流按位置回填控件数值，删除/重排会让旧画布
  错位（NaN 防御会兜底告警，但仍应避免）；弃用控件保留 no-op 默认值
- 新增输入/输出记得同步 `locales/{zh,en}/nodeDefs.json`（按输入名与输出索引组织，
  缺条目时回退英文名）
- 采样语义改动先读 `gh/long_video.py` 的 `run_long_video` / `run_refine_segments`
  与 `gh/sampling.py` 的双时钟实现；二跑调度的 T8 对齐依据写在节点 tooltip 与
  `run_refine_segments` docstring 里

## 致谢 / Credits

- 前身包 ComfyUI-MiniMaxH3-lanxi 的分段计划契约（`LANXI_H3_SEGMENTS`）与
- 两跑高清化精修配方对齐 T8 发布参数（simple 调度、入口去噪 0.30）

## 许可证 / License

[MIT](LICENSE)。
