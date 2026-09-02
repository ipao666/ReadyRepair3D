# Stage 6 — Final blind test / 最终盲测

The final test used 64 previously unused Chinese prompt groups. Base SANA and
the selected quality LoRA (`checkpoint-1500`) each generated four candidates
with identical seeds; the frozen Ready3D V3 selector chose one candidate per
group, followed by one Hunyuan3D reconstruction and eight-view quality scoring.

| Metric | Base SANA | Quality LoRA |
| --- | ---: | ---: |
| Mean final GLB Q | 0.47925 | 0.50876 |
| Qualified GLB rate | 1.56% | 4.69% |
| Prompt adherence | 83.33% | 79.17% |
| Technical-valid rate | 100.00% | 95.31% |

The paired bootstrap mean-Q difference was +0.02952, with a 95% CI of
[-0.01978, +0.07740]. The LoRA therefore did **not** meet the pre-registered
deployment gates: its adherence and technical-valid rates fell, and the paired
confidence interval includes negative benefit. The frozen decision is
`keep_base_sana`.

最终盲测使用 64 组此前从未参与训练或选择的中文提示词。Base SANA 与最终
LoRA（`checkpoint-1500`）使用完全相同的随机种子，各生成四张候选图；冻结的
Ready3D V3 为每组选择唯一 Top-1，再各运行一次 Hunyuan3D 和八视角质量评分。

LoRA 的平均最终 GLB 质量分更高（+0.02952），但 95% Bootstrap 置信区间跨越零，
且提示词遵循率下降 4.17 个百分点、技术有效率下降 4.69 个百分点。因此它未达到
预先注册的启用门槛，默认主链路保持 Base SANA；LoRA 作为明确的研究结果保留。

Machine-readable metrics: `final_comparison.json`.
