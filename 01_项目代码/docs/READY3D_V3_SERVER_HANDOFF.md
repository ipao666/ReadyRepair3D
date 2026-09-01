# Ready3D V3 Top-1 服务器两人交接单

## 目标与冻结边界

本轮只优化 Ready3D：从每组4张SANA候选图中选择唯一Top-1，再调用一次Hunyuan3D。不得修改 Ready3D V2、Hunyuan3D、Repair3D、既有PPT指标或冻结实验目录。所有选择权重只由56组训练集和12组验证集确定，12组测试集只运行一次最终评测。

固定数据协议为80组中文提示词、每组4个连续随机种子，共320张候选图；按提示词组隔离为56/12/12组。主输出根目录由 `R3D_V3_RUN_ROOT` 指定，默认是 `$R3DGUARD_HOME/data/ready3d_v3`。

## 开机后先做的共同检查

```bash
cd "$R3DGUARD_HOME"
git branch --show-current
git status --short
source activate.sh
python -m pytest -q
python ops/build_ready3d_v3_manifest.py \
  --prompts examples/ready3d_v3/prompts80.jsonl \
  --output-dir /tmp/ready3d_v3_manifest
python ops/run_ready3d_v3_server_preflight.py \
  --manifest /tmp/ready3d_v3_manifest/candidates320.jsonl \
  --output-root "${R3D_V3_RUN_ROOT:-$R3DGUARD_HOME/data/ready3d_v3}" \
  --expect 320
```

预检必须显示80组、320候选、固定种子有效。首次运行的320项下一阶段应为 `generate_2d`，且预检本身不会创建输出目录。

## 同伴A：生成负责人

目录所有权：

- `$R3D_V3_RUN_ROOT/manifests/`
- `$R3D_V3_RUN_ROOT/images/`
- `$R3D_V3_RUN_ROOT/hunyuan/`
- `$R3D_V3_RUN_ROOT/renders/`

唯一入口：

```bash
bash ops/run_ready3d_v3_generation.sh 2>&1 | tee "$R3D_V3_RUN_ROOT/logs/generation.log"
```

脚本依次完成Qwen中文提示词优化、固定种子SANA候选生成、320个Hunyuan带纹理GLB和每个GLB的8视角渲染。所有步骤可恢复；不要删除已有图片、GLB、状态文件或渲染图。完成条件：320张有效1024×1024图片、320个成功Hunyuan状态、每个资产8张渲染图。

## 同伴B：特征、评分与训练负责人

目录所有权：

- `$R3D_V3_RUN_ROOT/features/`
- `$R3D_V3_RUN_ROOT/auto_quality_v2/`
- `$R3D_V3_RUN_ROOT/splits/`
- `$R3D_V3_RUN_ROOT/training/`
- `$R3DGUARD_HOME/checkpoints/ready3d_v3_top1/`

同伴A生成完 `manifest.jsonl` 后即可先提取二维特征；等320个GLB和8视角完整后运行：

```bash
bash ops/run_ready3d_v3_scoring_training.sh 2>&1 | tee "$R3D_V3_RUN_ROOT/logs/scoring_training.log"
```

脚本会生成冻结V2自动标签、按56/12/12拆分数据、训练独立V3检查点，并输出Direct、Random Top-1、Structure Top-1、Ready V2 Top-1、Ready V3 Top-1、Oracle All-4六策略结果及95%组级Bootstrap置信区间。

## 严格交接条件

同伴A交给同伴B时必须提供 `manifest.jsonl`、`hunyuan/status.jsonl`、`renders/` 和 generation 日志。不得只交最终组选优CSV，因为训练需要320个候选的完整标签。现有Independent32摘要只保存组选优结果，不能用于补算V3 Top-1。

最终必须检查：

1. `preflight_complete.json` 中 `complete=320`；
2. V3 checkpoint schema 为 `r3dguard.ready3d-v3-top1-checkpoint.v1`；
3. metrics中 `test_used_for_selection=false`；
4. 每个非Oracle策略均为1次3D调用，Oracle仅表示4次调用的理论上限；
5. Ready V3 Top-1若未超过V2或随机基线，保留结果并报告，不得改测试阈值。

## 推荐显卡

优先租A100 40GB/80GB或L40S 48GB。Hunyuan3D纹理阶段历史峰值接近32GB，RTX 5090 32GB余量很小，只适合作为低价备选并降低并发；不要为了省少量费用反复迁移环境。
