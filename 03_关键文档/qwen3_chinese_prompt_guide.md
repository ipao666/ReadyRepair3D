# Qwen3 中文提示词线路使用说明

## 目标

把中文物体描述转换为适合 SANA 1.5 的英文提示词，同时严格保留主体、数量、颜色、材质、部件和结构。该模块独立于 independent32 主实验和 Repair3D，不修改二者的源码、权重或测试。

## 已实现线路

```text
中文提示词
  ↓
Qwen3-8B（非思考、贪心解码）
  ↓
结构化 JSON：英文主体描述 + 中文属性清单 + 置信度
  ↓
程序校验：原文回显、JSON Schema、颜色、材质、数量、冲突词
  ├─ 通过 → 固定追加 3D 重建约束
  └─ 失败 → 原中文 + 固定英文约束的安全回退，并记录原因
  ↓
SANA 1.5 Best-of-4（固定种子、可续跑）
```

Qwen 不负责自由美化，也不能修改固定构图约束。固定约束由代码追加，因此输出不受模型随机措辞影响。

## 文件

- `src/r3dloop/prompt_optimizer/core.py`：固定约束与属性冲突校验。
- `src/r3dloop/prompt_optimizer/qwen.py`：结构化输出解析、校验和回退。
- `src/r3dloop/prompt_optimizer/runtime.py`：Qwen3-8B 非思考模式运行时。
- `src/r3dloop/prompt_optimizer/batch.py`：稳定ID、JSONL断点续跑。
- `ops/optimize_chinese_prompts.py`：单条或批量中文提示词优化入口。
- `ops/generate_sana_from_optimized_prompts.py`：读取优化清单并生成每组4张候选图。
- `ops/download_qwen3_8b.sh`：镜像源、断点续传下载器。

## 输入格式

批量输入是 JSONL，每行至少有一个 `prompt` 字段：

```json
{"id":"demo_robot","prompt":"一个深红色黄铜机械猫，三条腿，齿轮结构"}
```

## 服务器命令

```bash
ssh r3dguard
cd /root/r3dguard
source activate.sh
export PYTHONPATH=/root/r3dguard/src:/root/r3dguard
```

查看模型下载：

```bash
PID=$(cat logs/qwen3_8b_download.pid)
kill -0 "$PID" && echo running || echo stopped
tail -f logs/qwen3_8b_download.log
du -sh models/Qwen3-8B
```

优化一条中文提示词（必须等 independent32 释放 GPU）：

```bash
python ops/optimize_chinese_prompts.py \
  --prompt '一个深红色黄铜机械猫，三条腿，齿轮结构'
```

批量优化：

```bash
python ops/optimize_chinese_prompts.py \
  --input examples/chinese_prompts.jsonl \
  --output data/chinese_prompts/optimized.jsonl
```

生成 SANA Best-of-4：

```bash
python ops/generate_sana_from_optimized_prompts.py \
  --input data/chinese_prompts/optimized.jsonl \
  --output-dir data/chinese_prompts/sana_bestof4
```

两个命令必须顺序执行。优化完成并退出后再加载 SANA，避免 Qwen 与 SANA 同时占用40GB显存。

## 输出字段

- `prompt_id`：根据内容生成的稳定ID，重复执行不变。
- `normalized_zh`：原中文原样放在开头，后接固定中文重建约束。
- `english_subject`：Qwen翻译的物体描述，不含相机和背景词。
- `sana_prompt`：英文主体描述加冻结的3D重建约束。
- `preserved_attributes`：数量、颜色、材质和结构的中文原词。
- `confidence`：Qwen自报置信度，当前门槛0.75。
- `validated`：模型翻译是否通过程序校验。
- `pipeline_ready`：是否可安全交给SANA；正常结果和安全回退均为 `true`。
- `used_fallback`、`fallback_reason`：是否回退及具体原因。

## 当前验证

- 新增线路测试：本地29项通过，服务器29项通过。
- 服务器相关回归组：47项通过，覆盖提示词、SANA清单、质量线路契约、Ready3D预测与验证集构建。
- Python编译检查通过。
- 全仓库直接运行会因已有 Repair3D 恢复包与根目录存在四个同名测试模块而产生 pytest 收集冲突；未删除或修改同伴目录。
- Qwen下载使用 `https://hf-mirror.com`，关闭不稳定的 Xet/hf_transfer 并保留标准断点续传。

## 下一步验收

1. 等模型权重下载完整并确认存在5个 safetensors 分片。
2. 等 independent32 释放 GPU 锁。
3. 用上面的机械猫提示词跑一次 Qwen，确认 `used_fallback=false`。
4. 生成4张 SANA 候选图，记录耗时、峰值显存和清单。
5. 再用10–20条中文提示词做小规模属性保真抽检；通过后才接入正式提示词入口。
