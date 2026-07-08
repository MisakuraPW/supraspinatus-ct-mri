# 方法记录

## 阶段一定位任务

当前任务定义为：在 CT 中自动生成冈上肌腱候选 ROI，而不是直接输出精细 tendon mask。

## 已落地路线

- 路线 A：骨阈值 + 连通域 + 固定物理尺寸 ROI 的规则法 baseline。
- 路线 B 轻量骨先验：用骨区域分布近似肩关节局部中心，生成候选 bbox/mask。
- 路线 C：已提供骨 mask、距离场和坐标通道生成脚本。
- 路线 D：已提供 SimpleITK 刚性配准弱标签迁移骨架。
- 路线 E：已提供从 ROI bbox 生成中心线热图的脚本。

## 预留路线

- 路线 B 完整版：肩胛骨/肱骨分割、关键点提取、局部坐标系和 oriented corridor。
- 路线 C 完整版：训练 CT + 骨 mask + 距离场 + 坐标通道网络，预测 tendon probability map。
