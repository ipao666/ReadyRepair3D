# 冻结发布清单

## 必需代码

- `src/r3dloop/`
- `ops/`
- `scripts/aggregate_final_results.py` / `ops/aggregate_final_results.py`
- `scripts/score_external64.py`
- `tests/`
- `examples/`
- `pyproject.toml` / `requirements*.txt` / `environment.yml`

## 冻结模型与配置

- `checkpoints/ready3d_v2/ready3d_v2.joblib`：当前部署。
- `config/quality_v2_calibration.json`：冻结质量阈值来源。

## 关键证据

- Independent32三策略汇总（Direct / Ready Top-2 / All）；Direct与Ready Top-2的组级结果及Bootstrap置信区间。
- Ready3D V2恢复指标。
- Repair3D安全路径案例摘要（30直接通过 / 2修改）。
- 中文茶壶最终端到端验收摘要。
- Showcase筛选报告和清单。

## 明确排除

- `models/`、`data/`、完整`evaluation/`、`repos/`、`logs/`、`tmp/`。
- `__pycache__/`、`.pytest_cache/`、锁文件、编辑器临时文件。
- `.env`、Hugging Face缓存Token、SSH凭据、个人配置。
- 批量GLB、批量渲染、视频和压缩展示包；仅保留`examples/showcase/`中的3个精选GLB及预览。

## 入口等级

1. `run_demo.sh` / `ops/run_full_quality_pipeline.sh`：完整质量闭环主入口。
2. `ops/run_repaired_quality_round.sh`：单轮Top-2 + 3D评分 + Repair3D。

