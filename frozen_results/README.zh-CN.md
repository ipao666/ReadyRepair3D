# 冻结实验记录

此目录保存作者历史实验的紧凑导出。[完整证据与限制](../docs/EVIDENCE.md)。

| 目录 | 仓库内实际内容 |
|---|---|
| `stage01_ready3d_v3_r2/` | 六种策略组级结果、汇总、校准与历史伪标签准入记录 |
| `stage02_sana_candidates/` | 生成摘要和哈希清单，不含批量候选图 |
| `stage05_lora_selection/` | 检查点选择与真实验证摘要 |
| `stage06_final_test/` | 64 组最终盲测摘要与 `keep_base_sana` 决策 |

Stage 3/4 的完成情况见最终报告，完整 checkpoint 和逐样本远程产物未导出。Stage 5/6 的远程 READY/哈希文件也未全部随仓库发布；清单可能引用未公开的远程文件。

V3 没有超过结构规则基线；最终 LoRA 的置信区间跨零，且遵循率与技术有效率下降。不得将记录写成显著或全面改善。

从仓库根目录运行 `python tools/verify_portfolio.py` 可重算 Stage 1 汇总和核验展示文件；缺少最终组级分数，因此不能独立重算最终 Bootstrap。
