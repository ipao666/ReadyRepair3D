# 第三方模型与恢复边界

本包只交付项目自研流水线、测试、小型检查点与实验摘要。第三方模型需根据各自许可证单独下载。

| 组件 | 用途 | 恢复信息 |
|---|---|---|
| SANA 1.5 1.6B | 4张候选图生成 | NVlabs/Sana；保持原始权重 |
| Qwen3-8B | 中文保守提示词优化 | 模型目录默认 `/root/r3dguard/models/Qwen3-8B` |
| DINOv2-L/14 | Ready3D语义与结构特征 | 模型目录默认 `/root/r3dguard/models/dinov2-large` |
| Depth Anything V2-L | 深度与边缘特征 | 模型目录默认 `/root/r3dguard/models/Depth-Anything-V2-Large-hf` |
| BiRefNet | 前景分割 | 模型目录默认 `/root/r3dguard/models/BiRefNet` |
| Hunyuan3D-2.1 | 主三维生成后端 | `https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git`，冻结提交 `82920d643c0dc2f7bfd7255f45f62d386edfe60c` |
| TRELLIS.2 | 辅助验证后端 | 不参与当前主链路与阈值校准 |

第三方权重不得被误认为本项目原创产物；重新分发前必须检查各仓库与模型页面的许可证条款。

