> 历史方案说明：本文中的 A/B、双人或队友描述属于当时的实施计划，不是实际贡献者名单。项目实际由 ipao666 独立完成；当前状态与运行入口以仓库首页为准。

# ReadyRepair3D V3 开发基线

记录日期：2026-08-27

## 1. 基线来源

- 冻结提交包：`C:\Users\17842\Desktop\ReadyRepair3D_提交包_20260726`
- 冻结代码：`C:\Users\17842\Desktop\ReadyRepair3D_提交包_20260726\01_代码`
- V3 开发目录：`C:\Users\17842\Desktop\ReadyRepair3D_V3`
- V3 初始提交：`e428aa45c73ec5eb053d43180b2d8f8b80d285b2`
- 冻结代码文件数：181
- 复制后逐文件 SHA256 不匹配数：0
- 冻结 `SHA256SUMS` 文件的 SHA256：`8F9D4D29D654BA7E8BB6051DCECDCE99ECC10487DC18FECDD847ACE3D14B0A50`

冻结提交包只作为 V2 对照，不在 V3 开发中修改。

## 2. 冻结实验指标

| 策略 | 平均 Q | 合格率 | 平均 3D 调用 | GPU 秒/组 |
|---|---:|---:|---:|---:|
| Direct | 0.6570 | 21.9% | 1 | 115.0 |
| Ready Top-2 | 0.7245 | 34.4% | 2 | 227.9 |
| All | 0.7495 | 47.2% | 4 | 457.2 |

Ready3D V2 测试规模为 6 个提示词组、24 张图片；绝对质量预测测试集 Spearman 为 `-0.1096`。该结果说明 V3 的首要任务应是提高组内 Top-1 排序能力，而不是继续依赖绝对质量回归。

## 3. 本机测试基线

测试环境：

- Windows PowerShell
- Python 3.13.7
- Git 2.53.0
- 测试时设置 `PYTHONPATH=src`
- 为 CPU 侧 GLB 测试补装 `trimesh==4.12.2` 与 `pygltflib==1.16.5`

执行命令：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
```

结果：

```text
199 passed, 1 skipped, 1 warning in 24.11s
```

唯一警告来自旧版 Hunyuan 运行时依赖的 `pkg_resources` 弃用提示，不属于当前代码功能失败。跳过项保持原项目测试条件，不在建立基线时改动。

## 4. V3 第一阶段目标

V3 优先把默认策略从 Ready Top-2 改进为单次 3D 生成的 Ready Top-1：

1. 使用提示词组级隔离的数据划分；
2. 以组内 Pairwise/Listwise Ranking 为主任务；
3. 保留绝对 Q 回归作为辅助任务；
4. 使用结构门、深度、法线、轮廓、语义约束和候选间相对特征；
5. 只允许增加低成本二维候选，不默认生成第二个 GLB；
6. 最终必须与 Direct、Random Top-1、Ready V2 Top-1、Ready V3 Top-1 和 All 比较。

在本地完成数据协议、排序代码、测试和批处理脚本前，不租用 GPU。

## 5. 服务器前实现状态（2026-08-27）

服务器前工作已在分支 `feature/ready3d-v3-top1` 完成：

- 80组中文提示词、56/12/12组级隔离与320候选固定种子协议；
- Qwen优化结果与固定候选ID/种子的无损衔接；
- 组内上下文特征、Pairwise Borda、质量回归和结构门集成；
- 每组严格选择一个候选，只计划一次Hunyuan3D调用；
- 低可信组只建议补充4张二维图，不恢复Top-2 GLB回退；
- 六策略测试集评测、组级Bootstrap置信区间和3D调用成本；
- 服务器只读预检、断点恢复计划与两人独立入口脚本。

本机最终回归为 `232 passed, 1 skipped`。实际生成、冻结V2复评分和新V3效果数字必须等GPU服务器完成，不使用旧Independent32组选优摘要伪造候选级结果。
