# 2026-07-02 杨国玲失败例复盘与修复

## 问题

在 `outputs/2026-07-02_unlabeled_generalized_inference` 的预览拼图中，第三行第一张对应病例 `杨国玲`。该例预测框明显落在肩胛冈/肩胛骨上方，而不是肱骨头附近，属于明显失败。

## 失败原因

该例不是 generalized 策略新引入的失败；legacy 和 generalized 的输出位置相同。

候选对比显示：

- rank 1 `low_z` 和 rank 21 `current_multibone` 的 ROI 位于肩胛骨上方。
- 这些候选的骨重叠、近骨比例、边界贴骨比例全为 0，因此被旧评分视为“很安全”。
- 但它们对应的 humerus anchor 实际上来自肩胛/关节盂附近的错误骨组件，属于伪肱骨锚点。
- rank 41/43/45 等 `surface_arc` 候选已经生成，并且位于肱骨头上方，更符合冈上肌腱包绕肱骨头的解剖位置。

换言之，这不是“没有生成正确候选”，而是“错误锚点候选因为完全避骨而得分过高”。

## 修复

在 `scripts/run_multibone_dicom_inference.py` 的 `generalized` selection policy 中新增跨锚点救援规则：

当主候选满足以下条件时，允许切换到其它锚点上的可信 `surface_arc` 候选：

- 主候选几乎完全离骨：`bone_overlap <= 0.001`、`near_bone_fraction <= 0.005`、`margin_bone_fraction <= 0.005`
- 主候选位于肱骨锚点顶端上方：`center_y_minus_humerus_top <= -2.0`
- 主候选锚点异常窄：`humerus_anchor_width <= 55`
- 存在分数接近、贴骨程度合理、锚点宽度更可信的 `surface_arc` 候选

该规则的意图不是强行贴骨，而是处理“伪肱骨锚点 + 过度避骨高分”的明显失败。

## 修复结果

单例 probe：

- 输出目录：`outputs/2026-07-02_unlabeled_generalized_rescue_probe`
- `杨国玲` 从 `current_multibone` 切换为 `surface_arc`
- 决策原因：`choose_generalized_cross_anchor_surface_rescue`
- 新中心：`(157.05, 212.99, 29.0)`
- 骨重叠：`0.0458`

完整 10 例：

- 输出目录：`outputs/2026-07-02_unlabeled_generalized_rescue`
- 成功病例：10
- 失败病例：0
- 平均骨重叠：0.0390
- 最大骨重叠：0.0484
- 最终来源：`surface_arc: 10`

与上一版 generalized 相比，只有 `杨国玲` 的最终候选发生改变，其余 9 例保持一致。

## 注意

该结果仍然是无标注数据上的视觉修复，不能等价于定量准确率提升。后续如果医生补标，应重新计算 center error、IoU 和 ROI coverage。
