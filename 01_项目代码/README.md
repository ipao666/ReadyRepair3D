> 本页主要记录历史 V2 工程链。当前项目总览、V3/LoRA 结果与轻量 CPU 入口见 [仓库首页](../README.zh-CN.md) 和 [复现指南](../docs/REPRODUCING.md)。这里的历史成绩与后续实验不可混用。

# ReadyRepair3D

面向高质量单图三维生成的智能筛选与闭环修复系统。
副标语：先判断值不值得生成，再决定如何安全修复。
发布口径：2026-07-22

## 1. 目录结构

```text
ReadyRepair3D/
├── run_demo.sh                 # 一键演示入口
├── activate_sana.sh            # 生图/特征环境 (conda: trellis2)
├── activate_hunyuan21.sh       # 三维/评分环境 (conda: hunyuan21)
├── activate.sh                 # 兼容旧名 → activate_sana.sh
├── verify_delivery.sh          # 交付测试
├── src/r3dloop/                # 中文提示词优化
├── ops/                        # 主链路与工具脚本
│   ├── _env.sh                 # 项目根目录与校准路径解析
│   ├── run_full_quality_pipeline.sh   # 主入口
│   ├── run_repaired_quality_round.sh
│   └── run_top2_dual3d_stage1.sh
├── scripts/                    # 汇总与外部打分工具
├── checkpoints/ready3d_v2/     # Ready3D V2 检查点
├── config/                     # 冻结质量阈值与校准
├── evaluation_summary/         # 主实验与验收摘要
├── examples/                   # 示例中文提示词
├── docs/                       # 设计与环境文档
├── environment/                # Conda/pip 环境快照
└── tests/                      # 自动测试
```

## 2. 主链路

```text
中文提示词 → SANA 4候选 → 结构门 + Ready3D Top-2
→ Hunyuan3D-2.1 → Auto3D 评分选优 → Repair3D safe_fallback → final.glb
```

环境变量（可选）：

- `R3DGUARD_HOME`：项目根目录（默认脚本自动检测）
- `R3DGUARD_MODELS`：模型目录（默认 `$R3DGUARD_HOME/models`）

## 3. 运行

先准备 Linux GPU 环境、Conda（`trellis2` / `hunyuan21`）、第三方模型与 Blender，参见 `docs/` 与 `THIRD_PARTY_MODELS.md`。

```bash
cd /path/to/ReadyRepair3D
bash run_demo.sh "一个蓝白陶瓷茶壶，完整主体，干净背景"
```

或：

```bash
bash ops/run_full_quality_pipeline.sh \
  --prompt "一个蓝白陶瓷茶壶，完整主体，干净背景" \
  --output-dir ./outputs/demo \
  --seed 20260722
```

主要输出：`final.glb`、`final_report.json`、`pipeline_summary.json`。

主入口会先用 Qwen3 对中文提示词做保守优化：严格保留主体、数量、颜色、材质和结构，仅补充适合单物体三维重建的构图约束；优化结果写入 `optimized_prompt.jsonl`，随后交给 SANA 生成候选图。

## 4. 主实验结果（Independent32）

- Direct：平均质量 `0.6570`，合格率 `21.9%`，GPU `115.0` 秒/组
- Ready Top-2：平均质量 `0.7245`，合格率 `34.4%`，GPU `227.9` 秒/组
- All：平均质量 `0.7495`，合格率 `47.2%`，GPU `457.2` 秒/组
- 合格阈值：`0.806912747446761`（`config/quality_v2_calibration.json`）

## 5. 测试

安装普通电脑侧的检查与测试依赖：

```bash
python -m pip install -r requirements-dev.txt
```

```bash
bash verify_delivery.sh
```

Windows可运行：

```powershell
.\verify_delivery.ps1
```

Windows或无GPU环境会跳过缺少第三方Hunyuan仓库或PyTorch Lightning运行时的集成检查；完整Qwen、SANA、Hunyuan3D和Blender链路仍需Linux GPU环境。GPU模型与仓库路径可通过`R3DGUARD_MODELS`、`HUNYUAN21_MODEL`、`HUNYUAN21_REPO`和`R3D_HUNYUAN_PYTHON`覆盖。

交付文件哈希由`ops/build_sha256s.py`跨平台生成和验证。

## 6. 示例成果

`examples/showcase/`包含茶壶、木雕大象和挖掘机三个真实带纹理GLB及预览图，来源和文件哈希记录在`examples/showcase_manifest.json`。

## 7. 未包含内容

- SANA / Qwen / DINO / Depth / BiRefNet / Hunyuan3D 等大型权重
- 原始图片、批量 GLB、八视角渲染与训练缓存
- 个人 Token / `.env` 凭据

## 8. 说明

正式修复路径为 `safe_fallback`。上交实验口径固定为 Direct、Ready Top-2、All 三种策略。
