# ReadyRepair3D 交接文件

更新日期：2026-09-02  
项目状态：主实验、LoRA 验证、64 组最终盲测均已完成并冻结。

## 一句话结论

质量加权 SANA LoRA 在最终 3D 质量上有正向趋势，但没有同时满足预注册的稳定性和统计门槛。因此当前**默认主链路保留 Base SANA**；LoRA `checkpoint-1500` 作为可复现实验成果保留，不应直接替换默认模型。

## 最终盲测结果（64 组独立中文提示词）

实验设计：Base SANA 与 LoRA 使用完全相同的 64 组未见提示词和随机种子；每组各生成 4 张候选，冻结 Ready3D V3 选 Top-1，随后各运行 Hunyuan3D、8 视角渲染及自动质量评分。

| 指标 | Base SANA | LoRA checkpoint-1500 | 解读 |
|---|---:|---:|---|
| 最终 GLB 平均质量 Q | 0.47925 | **0.50876** | LoRA +0.02952 |
| 合格 GLB 率 | 1.56%（1/64） | **4.69%（3/64）** | 小样本下提升 |
| 中文提示词遵循率 | **83.33%** | 79.17% | LoRA 下降 4.17pp |
| 技术有效率 | **100.00%** | 95.31%（61/64） | LoRA 下降 4.69pp |
| 配对 Bootstrap Q 差 95% CI | — | [-0.01978, +0.07740] | 区间跨 0，不能确认稳定正收益 |

预注册启用条件要求：平均 Q 提高、合格率相对提高至少 8%、遵循率下降不超过 3pp、技术有效率不下降、Bootstrap 不出现稳定负收益。LoRA 未通过后两项稳定性要求，决策为 `keep_base_sana`。

## 已完成阶段

| 阶段 | 状态 | 关键产出 |
|---|---|---|
| Stage 0 | 冻结 | 准备、提示词与测试协议 |
| Stage 1 | 冻结 | Ready3D V3 Top-1 |
| Stage 2 | 冻结 | SANA 候选与提示词遵循评分 |
| Stage 3 | 冻结 | 300 个 Hunyuan3D 真实质量标签 |
| Stage 4 | 冻结 | 质量加权 LoRA 训练、6 检查点代理筛选 |
| Stage 5 | 冻结 | 两检查点真实验证，选择 `checkpoint-1500` |
| Stage 6 | 冻结 | 64 组 Base vs LoRA 最终盲测 |

## 交付位置

- 私有代码仓库：<https://github.com/ipao666/ReadyRepair3D>
- 最终报告：[FINAL_REPORT.zh-CN.md](FINAL_REPORT.zh-CN.md)
- 最终机器可读指标：[frozen_results/stage06_final_test/final_comparison.json](frozen_results/stage06_final_test/final_comparison.json)
- Stage 5 选择指标：[frozen_results/stage05_lora_selection/selection_metrics.json](frozen_results/stage05_lora_selection/selection_metrics.json)
- 最近提交：`fc838b5`（最终报告）、`5ce0d5e`（Stage 5/6 结果与可复现修复）

远程 A100：

```bash
ssh matpool-a100
```

远程代码与结果：

```text
/root/ReadyRepair3D_V3
/root/readyrepair3d_runs/stages/stage05_real_validation
/root/readyrepair3d_runs/stages/stage06_final_test
/root/readyrepair3d_runs/final/final_comparison.json
```

每个冻结阶段内都有 `READY` 和 `SHA256SUMS`；如需验收：

```bash
cd /root/readyrepair3d_runs/stages/stage06_final_test
sha256sum -c SHA256SUMS
```

## 重要注意事项

1. 64 组最终测试已使用且已冻结，**不得再用于调参、筛选或重新训练**。
2. 若继续 LoRA 研究，必须新建隔离提示词测试集，重点提升中文属性遵循和 3D 技术稳定率。
3. 仓库不包含模型权重、候选图、GLB 和远程运行数据；它们体积大且仅保留在远程 A100。
4. Hunyuan 脚本已修复显式 `sample_id` 的优先级，避免多模型候选同名时覆盖输出。

## 建议的对外表述

> ReadyRepair3D 完成了从中文提示词、SANA 候选生成、Ready3D Top-1 选择到 Hunyuan3D 重建与自动质量评分的可恢复链路。质量加权 LoRA 在独立盲测中表现出平均 3D 质量提升趋势，但未达到预注册的稳定性阈值，因此系统当前保守地保留 Base SANA 为默认生成器，并将 LoRA 结论作为后续改进依据。
