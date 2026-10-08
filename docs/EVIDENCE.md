# 实验依据与已知限制

本页区分作者历史实验记录、可本地重算的结果与需要 GPU/原始产物才能重新验证的结论。自动 Q 是项目自定义指标，不是标准基准得分或人类评分；不同实验的数据与校准不应横向拼接。

## 三组实验

| 实验 | 样本与预算 | 证据 | 当前可验证范围 |
|---|---|---|---|
| Independent32 / V2 | 32 组；Direct 1 次、Top-2 2 次、All 4 次三维调用 | [摘要](../01_项目代码/evaluation_summary/independent32/main_experiment_report.md)、[组记录](../01_项目代码/evaluation_summary/independent32/strategy_group_results.jsonl) | Direct/Top-2 有组级记录；All 在此导出中仅有汇总，不能完整重算其统计 |
| Stage 1 / V3 | 12 组测试，每组 4 个候选；6 种策略 | [strategy_metrics.json](../frozen_results/stage01_ready3d_v3_r2/strategy_metrics.json) | 可重算六种策略的均值、遗憾值、Top-1 正确率与有效率；原始模型推理不在 CPU 核验范围 |
| Stage 6 / LoRA | 64 组新提示词；同种子，各 4 候选，V3 各选 1 个进行重建 | [最终比较](../frozen_results/stage06_final_test/final_comparison.json)、[最终报告](../FINAL_REPORT.zh-CN.md) | 可核对摘要算术与最终决策；未提交完整组级最终分数，无法从仓库独立重算 Bootstrap |

## 必须一起呈现的结果

- V3 平均 Q 为 0.5353，随机为 0.4546，结构规则为 0.5938；V3 相对 Direct 的差值置信区间跨零。
- V3 测试 Top-1 正确率为 0/12。“平均选中质量高于随机”不等于能可靠选出组内最优。
- 最终 LoRA 的 Q 差为 +0.02952，95% CI 为 [-0.01978, +0.07740]；合格资产仅由 1/64 变为 3/64。
- LoRA 提示词遵循率从 83.33% 降至 79.17%，技术有效率从 100% 降至 95.31%。遵循率指标不应擅自当作 64 个组上的二项计数。
- 冻结决策是 `keep_base_sana`。不能据此写“显著提升”“达到可商用质量”或“全面超过基线”。

## 冻结记录的范围

Stage 1 的训练器按验证集选择模型与融合权重。另一个历史 [pseudo-label gate](../frozen_results/stage01_ready3d_v3_r2/pseudolabel_gate.json) 把测试集优于随机作为后续伪标签准入条件，因此不能把“训练器没有用测试集选模”扩大成“整个后续研究流程从未参考测试结果”。最终 64 组测试是另一个独立评估阶段。

Stage 3/4 的完成情况由最终报告和历史交接记录记载，当前仓库没有完整训练 checkpoint 和逐样本远程产物。Stage 5/6 在仓库中是摘要导出；远程 `READY`/哈希记录不等于这些文件全部已公开。

## CPU CI 证明什么

`tools/check_cpu.py` 运行明确列出的轻量测试，再核验 Stage 1 记录、Stage 6 摘要算术及三份展示 GLB/预览的字节哈希。它不运行生成、LoRA 训练、Blender 渲染，也不验证论文级新颖性。模型与硬件结果需要在完整环境独立复跑。

## 后续可验证改进

优先比较相同三维调用预算下的结构规则与学习排序；增加按语义类别隔离的样本；补充有明确标注协议的人工质量核验；发布最终组级分数及完整环境清单。任何新模型选择都不能继续使用已冻结的 64 组最终测试集。
