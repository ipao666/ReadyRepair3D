> 历史方案说明：本文中的 A/B、双人或队友描述属于当时的实施计划，不是实际贡献者名单。项目实际由 ipao666 独立完成；当前状态与运行入口以仓库首页为准。

# SANA质量加权LoRA与Ready3D双人协作设计

日期：2026-08-27  
项目：ReadyRepair3D  
算力：1×A100 40GB  
方案：质量加权Flow-Matching LoRA

## 1. 目标与成功条件

本轮在不修改Gemma、DC-AE、Hunyuan3D和冻结Ready3D V2实验的前提下，只训练SANA 1.5 1.6B去噪Transformer的LoRA。训练监督由70%的校准3D下游质量和30%的中文提示词遵循度组成。

LoRA只有同时满足以下条件才允许进入主链路：

- 相对原始SANA，最终GLB合格率相对提升不少于8%；
- 最终GLB平均Q提高；
- 中文提示词遵循率下降不超过3个百分点；
- GLB技术有效率不下降；
- 64组全新测试提示词的组级Bootstrap结果不显示稳定负收益。

未达到门槛时保留LoRA为实验结果，主链路继续使用原始SANA。

## 2. 不可变边界

- 不训练或替换Gemma文本编码器与DC-AE。
- 不训练Hunyuan3D。
- 不恢复Ready Top-2默认回退；部署目标仍为一次Hunyuan3D调用。
- Ready3D V3冻结前，LoRA不得使用其预测分生成训练权重。
- 同一主体或提示词组不得跨训练、验证和测试划分。
- 测试集不用于选择LoRA检查点、融合权重或阈值。
- 两人不得直接修改对方拥有的源码、checkpoint或运行状态文件。

SANA官方Diffusers训练脚本作为实现基线：
`https://github.com/NVlabs/Sana/blob/main/train_scripts/train_dreambooth_lora_sana.py`。

## 3. 仓库与运行目录隔离

两人使用独立Git worktree，不在同一工作目录切换分支：

```text
/root/worktrees/ready3d-v3     队友B
/root/worktrees/sana-lora      你（成员A）
/mnt/r3dguard_shared           只存运行数据与交接件，不纳入Git
```

分支固定为：

```text
成员A：feature/sana-quality-lora
成员B：feature/ready3d-v3-top1
```

共享目录固定为：

```text
/mnt/r3dguard_shared/
├── inbox/                  # 上游提交待处理manifest
├── releases/               # 只读冻结交接件
├── locks/gpu0.lock         # A100互斥锁
├── logs/a/                 # 成员A日志
├── logs/b/                 # 成员B日志
└── scratch/                # 可删除中间缓存
```

任何交接件发布后不可原地修改；修正时创建递增版本目录，例如 `ready3d_v3_r2/`。

## 4. 源码所有权

### 成员A（你）：SANA与提示词遵循

独占以下新目录：

```text
src/r3dloop/sana_lora/
ops/sana_lora/
tests/sana_lora/
configs/sana_lora/
docs/sana_lora/
```

职责：

1. 建立240组全新中文提示词，按180/30/30组划分。
2. 调用既有Qwen提示词优化器，生成英文SANA caption，同时保留中文原文。
3. 用原始SANA为每组生成4张固定种子候选图，共960张。
4. 运行Qwen3-VL-8B提示词遵循评分，输出主体、数量、颜色、材质、部件和结构六项分数。
5. 接收成员B发布的冻结Ready3D/Hunyuan标签，计算70/30综合权重。
6. 训练rank 16 LoRA，保存每250步检查点，最多1500步。
7. 生成原始SANA与LoRA的固定种子验证、测试候选图。

成员A不得修改：

```text
src/r3dloop/ready3d_v3/
ops/train_ready3d_v3_top1.py
ops/run_hunyuan_*.py
ops/score_hunyuan_outputs.py
checkpoints/ready3d_v3_top1/
```

### 成员B（队友）：Ready3D、Hunyuan与质量标签

独占以下目录：

```text
src/r3dloop/ready3d_v3/
ops/*ready3d_v3*.py
tests/test_*ready3d_v3*.py
checkpoints/ready3d_v3_top1/
evaluation/ready3d_v3/
```

职责：

1. 完成80组、320候选的Ready3D V3真实标签生成和训练。
2. 冻结V3 checkpoint、特征名、结构门、融合权重和验证阈值。
3. 接收成员A的LoRA候选manifest，运行Hunyuan3D、8视角渲染和冻结V2自动Q评分。
4. 为LoRA训练集发布真实Hunyuan分和Ready3D校准分。
5. 对原始SANA与LoRA的最终测试候选执行相同Hunyuan流程。
6. 输出候选级标签；不得只输出组级汇总。

成员B不得修改：

```text
src/r3dloop/sana_lora/
ops/sana_lora/
configs/sana_lora/
checkpoints/sana_lora/
```

## 5. 单卡GPU互斥规则

所有会初始化CUDA的命令必须持有同一把文件锁：

```bash
flock /mnt/r3dguard_shared/locks/gpu0.lock bash -lc '<GPU命令>'
```

没有拿到锁的任务等待，不得绕过锁启动第二个CUDA进程。CPU数据校验、JSONL整理、测试、报告和哈希计算不需要GPU锁。

独立环境固定为：

```text
sana-lora：SANA、Diffusers、PEFT、Qwen3-VL
ready3d：DINOv2、Depth Anything、BiRefNet、scikit-learn
hunyuan21：Hunyuan3D-2.1与纹理模块
```

任何人不得在另一个人的环境中执行 `pip install`。

## 6. 数据交接协议

每个release目录必须包含：

```text
manifest.jsonl
summary.json
SHA256SUMS
READY
```

`READY`最后原子写入。下游只读取存在 `READY` 且 `SHA256SUMS` 验证通过的版本。

### B→A：冻结Ready3D

目录：

```text
/mnt/r3dguard_shared/releases/ready3d_v3_r1/
├── ready3d_v3_top1.joblib
├── metrics.json
├── calibration.json
├── feature_contract.json
├── SHA256SUMS
└── READY
```

`metrics.json`必须声明 `test_used_for_selection=false`。A验证通过后才能计算LoRA样本权重。

### A→B：LoRA候选标注请求

目录：

```text
/mnt/r3dguard_shared/inbox/sana_lora_candidates_r1/
├── manifest.jsonl
├── images/
├── adherence_scores.jsonl
├── SHA256SUMS
└── READY
```

manifest必须保存 `sample_id`、`prompt_group_id`、`split`、中文原提示词、英文caption、seed、图片路径和base model版本。

### B→A：LoRA训练标签

目录：

```text
/mnt/r3dguard_shared/releases/sana_lora_labels_r1/
├── candidate_labels.jsonl
├── calibration.json
├── coverage.json
├── SHA256SUMS
└── READY
```

每条标签同时标明真实Hunyuan Q或Ready3D伪标签来源；不得混为同一种ground truth。

### A→B：最终对照候选

目录分别为：

```text
/mnt/r3dguard_shared/inbox/final_base_sana_r1/
/mnt/r3dguard_shared/inbox/final_lora_sana_r1/
```

两套候选必须使用完全相同的64组提示词和随机种子。

## 7. 样本、权重与训练配置

240组新提示词划分为180/30/30组，每组4张，共960张。训练集720张全部运行冻结Ready3D V3，并从中按质量分位数、类别和风险类型分层抽取180张运行真实Hunyuan。验证集120张全部运行Hunyuan。测试集120张在训练结束前不评分。

Qwen3-VL对全部960张图产生0到1的提示词遵循分。真实Hunyuan标签优先；其余训练样本使用验证集校准后的Ready3D分：

```text
combined_quality = 0.70 * calibrated_3d_quality + 0.30 * prompt_adherence
sample_weight = clip(0.25 + 1.25 * combined_quality, 0.25, 1.50)
```

训练参数冻结为：

```text
base model       SANA 1.5 1.6B 1024px
trainable        transformer LoRA: to_q,to_k,to_v
rank/alpha       16/16
precision        BF16
batch            1
gradient accum.  8
learning rate    5e-5
optimizer        8-bit AdamW
steps            1500
warmup           100
checkpoint       every 250 steps
grad clip        1.0
crop/flip        disabled
checkpointing    enabled
text/DC-AE       frozen and offloaded when unused
```

## 8. 无冲突执行顺序

### 阶段0：不开服务器时

- A完成LoRA代码、240组提示词、数据schema和CPU测试。
- B维护已完成的Ready3D V3代码、80组提示词和服务器预检。
- 两人不得共同修改README、总报告或PPT；最终由A在合并分支统一更新。

### 阶段1：先冻结Ready3D V3

- B独占GPU，完成320候选的Hunyuan标签、V3训练和六策略评测。
- A同时只做CPU任务：审核240组提示词、检查数据隔离、准备LoRA trainer和评测模板。
- 阶段门：B发布 `ready3d_v3_r1/READY`。

### 阶段2：生成LoRA数据

- A独占GPU，生成960张原始SANA候选并运行Qwen3-VL评分，发布候选请求。
- B不启动GPU任务，只校验manifest、准备Hunyuan恢复队列。
- 阶段门：A发布 `sana_lora_candidates_r1/READY`。

### 阶段3：生成下游标签

- B独占GPU，运行300个Hunyuan标注任务：180张分层训练样本和120张验证样本。
- A只做CPU数据检查，不得在标签未冻结时训练。
- 阶段门：B发布 `sana_lora_labels_r1/READY`。

### 阶段4：训练LoRA

- A独占GPU训练1500步，并用低成本Ready3D与Qwen3-VL筛选6个检查点中的前2名。
- B只审查训练manifest与权重计算，不修改LoRA文件。
- 阶段门：A发布候选checkpoint及其SHA256。

### 阶段5：真实验证与最终测试

- A为两个候选checkpoint生成验证图；B分别对32组运行Hunyuan，选定最终LoRA。
- A随后生成原始SANA和最终LoRA的64组固定种子测试候选。
- B运行相同的Hunyuan、渲染和冻结Q评分，输出候选级结果。
- A负责只读汇总报告；任何测试结果不得反向改变训练配置。

## 9. 故障与回退

- GPU任务中断：保留状态文件，恢复缺失sample ID，不整批重跑。
- 交接哈希失败：拒绝消费，发布方创建新版本目录，不覆盖旧目录。
- Ready3D V3未超过V2或随机Top-1：仍可完成LoRA真实Hunyuan子集训练，但禁止把未经验证的V3伪标签扩展到720张训练图。
- LoRA未达到启用门槛：最终系统继续加载原始SANA，LoRA作为实验与后续偏好训练素材。
- Qwen3-VL评分解析失败：该样本权重只使用3D质量，记录 `adherence_missing=true`，不填造分数。

## 10. 预计资源

单张A100 40GB预计总GPU时间21至25小时：SANA与Qwen3-VL约3小时，300个Hunyuan标签约10小时，LoRA约4至6小时，验证和测试Hunyuan约4至5小时。建议共享存储至少250GB。

该时间不包含首次模型下载和环境安装。所有GPU阶段严格串行，CPU准备阶段可以并行。
