# Repair3D 最小可靠规则版

## 结论

当前默认路径是确定性 `safe_fallback`，不加载、不恢复、不训练 Repair3D V2/V3 学习模型。

系统执行：

1. 诊断 GLB 的连通组件、重复顶点、退化/重复面、法线、小孔、材质、纹理和 UV；
2. 只尝试低风险动作；
3. 重新加载并检查几何、纹理和输入一致性；
4. 在 A100 上完成 8 视角冻结质量复评分；
5. 仅当全部安全门通过且 `Q_after >= Q_before + 0.005` 时输出 `repair_accepted`；
6. 其余情况输出原始 GLB，并验证 `SHA256(final) == SHA256(original)`。

## 允许的动作

- 删除总面积或总面数占比低于 1% 的非主体漂浮组件；
- 合并坐标极近的重复顶点；
- 删除退化面和重复面；
- 修正反向或不一致法线；
- 仅填补最多 2 个、边界边总数不超过 16 的小孔。

默认禁止整体重网格、大规模补洞、强力减面和完整重纹理。

## 状态

- `accept_original`：资产健康，无需修复；
- `repair_accepted`：修复候选通过全部安全门和冻结 Q 增益门；
- `repair_rollback`：尝试修复但未证明安全提升，最终 SHA 一致回退；
- `regenerate_required`：主体严重缺失、大面积开放边界或灾难性拓扑，建议重新生成。

## 本机运行

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python ops/run_repair3d_auto.py `
  "C:\path\input.glb" `
  --output-dir "evaluation\repair_case" `
  --blender "D:\blender\blender.exe"
```

输出包括 `original.glb`、`candidate.glb`、`final.glb`、`repair_report.json`
和 `pipeline_summary.json`。

## A100 完整闭环

`ops/run_repaired_quality_round.sh` 已改为对 `candidate.glb` 做 8 视角渲染和冻结
V2 复评分。`ops/finalize_repaired_round.py` 负责应用 0.005 增益门、更新
`repair_report.json`，并在拒绝时进行 SHA256 一致回退。

## 望远镜真实案例

输入：`C:\Users\17842\Desktop\项目\失败\望远镜.glb`

输出：`evaluation/repair3d_minimal_rule_telescope_20260726/`

诊断结果：

- Trimesh 与 Blender 均可加载；
- 25,570 个顶点，40,000 个面，全部有限；
- 1 个连通主体，最大主体占比 1.0；
- 0 个开放边、非流形边、退化面和重复面；
- 材质、2 张纹理图和 UV 均有效；
- 未发现允许动作能够安全改善的几何缺陷。

因此状态为 `accept_original`。`original.glb`、`candidate.glb` 和 `final.glb`
的 SHA256 均为：

`d902a7feadc6332b25529cd2b1bd6a081b60f16ec974d3b3d4c2e926d34909f0`

这表示该文件在规则允许范围内无需修复；视觉语义或造型问题应通过重新生成解决，
不应通过高风险网格操作强行修改。

## 测试

最终本机测试结果：`198 passed, 2 skipped`。跳过项为环境相关测试，不影响
Repair3D 规则路径；另有一个第三方 `pkg_resources` 弃用警告。
