# 当前维护入口

作者与维护者：**ipao666（独立个人项目）**。

当前入口是 [README.zh-CN.md](README.zh-CN.md)、[复现指南](docs/REPRODUCING.md) 和 [证据说明](docs/EVIDENCE.md)。历史交接方案不作为求职作品的贡献者名单或当前安装指南。

工程演示保留 V2 Top-2 + Base SANA + safe_fallback。V3 Top-1 与质量加权 LoRA 是独立研究路径。最终 64 组 LoRA 盲测的均值差置信区间跨零，遵循率和技术有效率下降，因此冻结决策为 `keep_base_sana`。

- [最终报告](FINAL_REPORT.zh-CN.md)
- [Stage 5 选择记录](frozen_results/stage05_lora_selection/selection_metrics.json)
- [Stage 6 最终比较](frozen_results/stage06_final_test/final_comparison.json)

最终 64 组测试集不得继续用于调参。完整远程模型与实验产物不在仓库内；三份展示 GLB 及预览已提交，并可进行哈希核验。
