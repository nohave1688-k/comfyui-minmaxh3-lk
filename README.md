# ComfyUI MiniMax H3 Long Video

> MiniMax H3 长视频工作流插件 —— ComfyUI-MiniMaxH3-lanxi 的瘦身重建版。
>
> 只保留长视频核心链路，节点 ID 全新命名，可与旧 lanxi 包共存。

## 节点

### LongvideoH3Studio —— MiniMaxH3长视频操作台（longvideo）

条件化构建器 + 分段面板：

- 三种模式：文生关键帧 / 全参考 / **多分段长视频**
- 1-8 个分段，每段独立的提示词、首帧/尾帧、图片/视频/音频参考、RefMod、LoRA 槽（×4）
- **按分段连线媒体**（本插件的增量功能）：把上游节点（加载图片/加载视频/加载音频/PS）的连线
  直接拖到面板按钮上即可——连线内容覆盖面板上传的同槽位素材；按钮实时显示连接状态，
  点击已连线按钮断开并恢复面板素材
- 分段运行范围、复制上一段、提示词标签（`<图片n>` `<视频n>` `<音频n>`）、提示词优化器

输出 `segments` 分段计划，接长视频分段采样。

### LongvideoH3LongVideo —— MiniMax-H3 长视频分段采样（longvideo）

分段采样执行器：逐段 LoRA 挂载 → 双时钟采样（视频/音频独立调度）→ 段间 latent 续接
（上一段尾部 7 个 latent token / 22 帧无缝衔接）→ 音视频联合解码 → 拼接。
参数：steps/shift_video/shift_audio/seed(+策略)/context_noise（接缝噪声）/cfg/release_clip/free_between_segments。

## 与旧 lanxi 包的关系

| | lanxi（旧） | 本包 |
|---|---|---|
| 节点 | 7 个已注册 + 16 个未注册 RefMod | **2 个**（纯长视频核心） |
| 注册方式 | V1 双轨 + comfy_entrypoint | 纯 `NODE_CLASS_MAPPINGS`（V3 类原生执行） |
| 节点 ID | `LanxiH3Studio` 等 | `LongvideoH3Studio` 等（可共存） |
| RefMod 库 | `models/refmods/` | 同一路径，**两包共享** |
| API 路由 | `/lanxi/...` | `/longvideo/...` |

移除内容：单模式管线（适配器 + 独立双时钟采样 + AV 解码三件套，上游已断链）、
二采分块采样器、Latent 上采样器、16 个未注册的 RefMod 图节点（RefMod 库功能保留在面板底层）。

**注意**：两包可同时安装共存（节点 ID 不同），但请勿同时把两家的面板用于同一个
工作流节点；迁移旧工作流时，把 `LanxiH3Studio` 换成本包的 `LongvideoH3Studio` 并重新接线。

## 安装

放入 `ComfyUI/custom_nodes/`，重启 ComfyUI。依赖与 lanxi 相同（见 requirements.txt）。

## 结构

```
gh/            引擎（操作台、分段计划、双时钟采样、条件化、音频解码、提示词优化）
refmod/        RefMod 参考库（面板底层，存储于 models/refmods/）
web/js/        操作台前端面板
locales/       中英文翻译
```

## 增量开发

- 分段数据契约：`_normalize_segment`（gh/long_video.py）与前端 `normalizeSegment` 需同步
- 连线媒体：插座名 `conn_segN_kind.N`（N 为 0 起始序号），收集逻辑在 `gh/nodes.py::_collect_connected_media`
- API 路由前缀 `/longvideo/...`（loras / prompt-optimizer / refmods/library）
