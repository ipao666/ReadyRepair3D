# 复现与演示指南

所有命令从仓库根目录执行，除非明确标注。推荐 Python 3.11。CPU 核验、模型推理和历史实验复现具有不同依赖及证据范围。

## 1. 无依赖记录核验

```bash
python tools/verify_portfolio.py
python -m unittest discover -s tools -p 'test_*.py'
```

只使用 Python 3.10+ 标准库。核对六种策略的组覆盖、均值、遗憾值、同资产评分一致性、最终摘要算术及展示文件 SHA-256。输出明确标注最终 Bootstrap 未被重算。改变源文件或记录不一致会返回非零退出码。

## 2. 轻量 CPU 测试

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 使用 .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt
python tools/check_cpu.py
```

测试文件显式列在 `tools/check_cpu.py`。这个配置避开 PyTorch、SciPy、模型权重和 Blender，只覆盖数据协议、排序运算、质量聚合、LoRA 权重/配置及证据一致性。它不替代原工程完整测试。

## 3. 完整工程与模型运行

完整工程位于 `01_项目代码/`。其 `requirements-dev.txt` 是历史工程依赖集合，包含重型视觉库；不是上面的轻量 CPU 配置。完整测试需要相应依赖，部分集成检查另需模型和第三方仓库。

```bash
cd 01_项目代码
python -m pip install -r requirements-dev.txt
python -m pytest tests -q --import-mode=importlib
```

模型运行需 Linux GPU 环境、第三方模型和 Blender。按 [模型来源与许可证](../01_项目代码/THIRD_PARTY_MODELS.md) 和 [环境指南](../01_项目代码/SERVER_SETUP.md) 配置，不把旧文档中的个人服务器路径照搬为本机路径。

```bash
cd 01_项目代码
export R3DGUARD_MODELS=/absolute/path/to/models
# 配好 activate_sana.sh / activate_hunyuan21.sh 所需环境后
bash run_demo.sh "一个蓝白陶瓷茶壶，完整主体，干净背景"
```

该演示是历史 V2 Top-2 工程链，默认 Base SANA 与 safe_fallback。其输出包括 GLB、评分与流水线报告。它不是 V3/LoRA 64 组盲测的“一键复跑”；后者还依赖未公开的训练/生成产物。硬件预算应依据本机模型预检确认。

## 4. 历史文件怎么读

- `FROZEN_RELEASE.json` 与根目录 `SHA256SUMS_PACKAGE.txt` 记录历史交付或迁移快照，不能作为当前 Git HEAD 清单。代码目录的 `SHA256SUMS` 已随当前代码更新，可用 `ops/build_sha256s.py --verify` 检查；这不改变历史实验结果。
- `01_项目代码/verify_delivery.sh`（或 `.ps1`）默认运行轻量测试和代码文件哈希核验；`--full`（PowerShell 为 `-Full`）运行完整工程测试。
- `verify_package.ps1` 是兼容入口，现在执行当前作品证据核验；它不会声称验证已不完整公开的迁移包。
- `03_关键文档/` 与旧设计文档中的“双人/A/B”描述是历史计划；项目实际由 ipao666 独立完成。
- 实验 JSON、展示资产及其来源哈希保持原样。作品整理没有新增实验成绩，也没有把旧汇总重新标记成新实验。

若全量测试出现 SciPy 二进制或 CUDA/Blender 依赖错误，应使用隔离且匹配的完整环境。不要把跳过集成测试当成模型结果验收。
