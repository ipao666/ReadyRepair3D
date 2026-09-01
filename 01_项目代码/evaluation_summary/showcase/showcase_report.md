# ReadyRepair3D 明日汇报展示筛选报告

## 审计概况

- 递归找到 GLB：1803 个
- 排除训练故障注入、动作试验、fixture或外部示例：1383 个
- 来源明确、可加载、有嵌入纹理且拓扑未触发严重阈值：395 个
- 最终入选：8 个；全部通过 Trimesh 加载，并由 Blender 统一重渲染四个视角后人工复核。
- 自动分只用于候选排序；最终选择同时检查语义、背面体积、纹理、碎片、孔洞和展示类别多样性。

## 最推荐现场展示

1. **01_blue_white_teapot**：`/root/r3dguard/evaluation/final_acceptance_20260722/pipeline/final.glb`，Q=0.8222，预览：`/root/r3dguard/showcase_results/cases/01_blue_white_teapot/four_view_montage.png`
2. **03_wooden_elephant**：`/root/r3dguard/data/independent32/hunyuan/glb/ind32_024_c1_s2026072341.glb`，Q=0.9270，预览：`/root/r3dguard/showcase_results/cases/03_wooden_elephant/four_view_montage.png`
3. **04_toy_excavator**：`/root/r3dguard/data/chinese_prompt_smoke/hunyuan_selected7/glb/zh_demo_003_c3_s2031136393.glb`，Q=0.8109，预览：`/root/r3dguard/showcase_results/cases/04_toy_excavator/four_view_montage.png`

## PPT案例（8个）

|排名|类别|策略|Q|组件|最大主体占比|已知缺陷|
|---:|---|---|---:|---:|---:|---|
|1|vessel / ceramic|Final full chain: Ready Top-2 + Hunyuan3D + Repair3D safe-fallback accept|0.8222|1|1.000000|局部高光贴图略有斑块，背面把手轮廓比输入略厚。|
|2|animal / decoration|Independent32 All / Best-of-4 (visual showpiece; not a Ready-selection claim)|0.9270|2|0.999995|背部雕刻细节较正面简化，两侧象牙略不完全对称。|
|3|vehicle / machinery|Chinese prompt best-of-4 + Hunyuan3D|0.8109|6|0.998388|原始组件数为6，机械臂关节和铲斗内侧细节有简化。|
|4|electronics / mechanical|Independent32 Ready Top-2 + Repair3D safe-fallback accept|0.9170|2|0.999952|背板较光滑，铃铛连接和支脚局部略有融合。|
|5|furniture|Chinese prompt best-of-4 + Hunyuan3D|0.8058|1|1.000000|布料缝线和木腿材质细节偏少，靠背背面较简化。|
|6|vehicle / toy|Independent32 direct / All-best visual result|0.8764|1|1.000000|翼间支撑和尾部结构偏简化，背面视角显得稍薄。|
|7|utensil / decoration|Independent32 Ready Top-2 + Repair3D safe-fallback accept|0.8656|1|1.000000|手柄花纹略不对称，钟舌在部分角度被遮挡。|
|8|creative / toy|Independent32 direct / All-best visual result|0.8650|3|0.999930|触手底部与背面纹理较简化，原始组件数为3但碎片占比极低。|

## 对比图

对比图严格标注了Direct/基线与Ready所使用的不同2D候选，避免把候选选择收益误写成同一输入下的3D修复收益。

- `/root/r3dguard/showcase_results/comparisons/01_teapot_full_chain.png`：Q 0.8136 -> 0.8222
- `/root/r3dguard/showcase_results/comparisons/02_handbell_ready_gain.png`：Q 0.3647 -> 0.8656
- `/root/r3dguard/showcase_results/comparisons/03_projector_ready_gain.png`：Q 0.4123 -> 0.7340
- `/root/r3dguard/showcase_results/comparisons/04_rice_cooker_ready_gain.png`：Q 0.8431 -> 0.9157

## 不建议展示

- **final_teapot_baseline** (`request_2026072201_c0_s2026072201`)：自动分0.8136但出现大面积平面背景几何，说明不能只看分数。
- **elephant_ready_c2** (`ind32_024_c2_s2026072342`)：长尾状几何拉大包围盒，主体在统一视图中过小且侧后轮廓异常。
- **typewriter** (`ind32_028_c1_s2026072381`)：机身、键盘和纸架明显融合变形，机械结构不可读。
- **telescope** (`ind32_030_c3_s2026072403`)：三脚架和镜筒比例失真，支撑结构不完整。
- **field_recorder** (`ind32_007_c2_s2026072172`)：整体更像遥控器，交叉麦克风和控制面板语义缺失。
- **mechanical_cat** (`zh_demo_001_retry2_c2_s2028556752`)：颜色与黄铜语义偏差大，三腿要求和背面机械细节不稳定。

## 关键文件

- 总览：`/root/r3dguard/showcase_results/showcase_master_contact.png`
- 清单：`/root/r3dguard/showcase_results/showcase_manifest.csv`
- 全库审计：`/root/r3dguard/showcase_results/all_glb_inventory.csv`
- 对比总览：`/root/r3dguard/showcase_results/comparisons/comparison_index.png`
- 失败案例：`/root/r3dguard/showcase_results/not_recommended_contact.png`
