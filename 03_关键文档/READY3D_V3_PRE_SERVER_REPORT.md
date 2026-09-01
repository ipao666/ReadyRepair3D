# Ready3D V3 Top-1 服务器前完成报告

日期：2026-08-27  
开发目录：`C:\Users\17842\Desktop\ReadyRepair3D_V3`  
分支：`feature/ready3d-v3-top1`

## 结论

不开GPU服务器能完成的工作已经完成。代码现在具备从80组中文提示词开始，经Qwen优化、固定种子SANA候选、候选级Hunyuan标签、组内Top-1训练，到六策略测试评测的完整可恢复流程。默认目标已从两次3D生成的Ready Top-2改为一次3D生成的Ready V3 Top-1。

当前不能声称V3质量已经提高，因为本机没有SANA/Hunyuan运行环境，也没有320个新候选的GLB与冻结V2标签。现有Independent32提交摘要只包含Direct、Ready Top-2和All的组选优结果，不含128个完整候选级标签，不能用于训练或补算V3 Top-1。

## 已完成内容

1. **数据协议**：80个互不泄漏的中文提示词组，56组训练、12组验证、12组测试；每组4个固定连续种子，共320个候选。
2. **中文提示词链路**：Qwen3-8B优化结果使用原group ID回填，SANA生成阶段严格保留预先冻结的sample ID、candidate index和seed。
3. **Top-1模型**：候选特征加入组内中心化和标准化上下文；成对排序、绝对质量回归与结构门集成；集成权重只用验证集选择。
4. **单次3D推理**：每组输出且只输出一个 `selected_for_3d=true`；结构全失败或预测质量过低时只建议再采4张二维图，`planned_3d_calls`仍为1，Top-2回退关闭。
5. **评测**：支持Direct、Random Top-1、Structure Top-1、Ready V2 Top-1、Ready V3 Top-1、Oracle All-4；输出平均Q、合格率、技术有效率、Top-1命中率、遗憾值、质量捕获率、3D调用和相对Direct的95%组级Bootstrap区间。
6. **恢复与交接**：预检能够识别每个样本下一阶段是生成二维图、生成3D、渲染、评分或完成；服务器两人目录所有权和命令已经分开。

## 本地验证证据

- 完整测试：`232 passed, 1 skipped, 1 warning`。
- Python脚本语法编译：通过。
- 实际清单：80组、320候选、56/12/12、10个类别、8类风险覆盖。
- 固定种子：从 `2026082700` 开始，每组4个连续种子，320个sample ID无重复。
- 只读预检：`ready=true`，初始 `generate_2d=320`，其余阶段为0。
- 训练冒烟：合成组级夹具能够训练独立checkpoint，并保证测试集不参与模型与权重选择。

本机 `bash.exe` 指向损坏的WSL入口，因此没有假装完成Linux Bash执行验证。服务器首次同步后应运行：

```bash
bash -n ops/run_ready3d_v3_generation.sh
bash -n ops/run_ready3d_v3_scoring_training.sh
```

## 服务器任务分工

### 同伴A：生成

运行 `bash ops/run_ready3d_v3_generation.sh`。负责Qwen优化、320张SANA图、320个Hunyuan带纹理GLB和2560张固定视角渲染图。只写 `manifests/`、`images/`、`hunyuan/`、`renders/`。

### 同伴B：评分与训练

运行 `bash ops/run_ready3d_v3_scoring_training.sh`。负责二维特征、冻结V2自动标签、训练/验证/测试物理拆分、V3训练和六策略评测。只写 `features/`、`auto_quality_v2/`、`splits/`、`training/` 和V3 checkpoint目录。

详细命令、环境变量和交接门槛见 `docs/READY3D_V3_SERVER_HANDOFF.md`。

## 租卡后必须取得的判定结果

Ready V3 Top-1首先应比较Ready V2 Top-1和Random Top-1，而不只比较Direct。推荐启用门槛：

- 平均Q高于Ready V2 Top-1，且相对Random Top-1的95%置信区间不明显为负；
- 合格率不下降；
- 每组保持1次Hunyuan3D调用；
- 所有阈值和集成权重在查看测试结果前冻结。

若V3未超过随机或V2，应如实保留结果，分析困难类别，不在测试集上重新调权重。Oracle All-4只表示理论上限，不能作为可部署方案。
