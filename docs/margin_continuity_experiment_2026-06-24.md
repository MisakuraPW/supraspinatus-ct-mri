# 骨距离 Margin 与 Z 层连续性实验 2026-06-24

## 1. 本轮目标

在老师传统方法和现有多骨锚点方法基础上，继续优化两个问题：

1. ROI 不仅不能包含骨体素，还应避免边界 1-3 voxel 内贴骨。
2. 候选不能只由单层骨组件决定，应要求相邻 3-5 层存在稳定的肱骨-顶板结构。

## 2. 代码改动

修改文件：

- `src/supraspinatus_locator/localization/multi_bone_traditional.py`
- `scripts/run_multibone_locator.py`

新增逻辑：

- `bone_shell_fraction(...)`：统计 ROI 外扩壳层内的骨体素比例。
- `near_bone_fraction`：ROI 外扩 1 voxel 的近邻骨比例。
- `margin_bone_fraction`：ROI 外扩 `bone_margin_voxels` 的壳层骨比例，默认 3。
- `anchor_continuity_score(...)`：检查相邻 z 层内是否存在相近的肱骨-顶板候选。
- 候选评分新增：
  - `+ continuity_score * 0.55`
  - `- near_bone_fraction * 3.0`
  - `- margin_bone_fraction * 1.4`
- `run_multibone_locator.py` 新增参数：
  - `--bone-margin-voxels`
  - `--continuity-window`
  - `--continuity-xy-tolerance`

默认参数：

```bash
python scripts/run_multibone_locator.py \
  --data-dir outputs/normalized_10cases \
  --output-dir outputs/multibone_10cases_margin_continuity \
  --bone-margin-voxels 3 \
  --continuity-window 2 \
  --continuity-xy-tolerance 14
```

## 3. 总体结果

| 方法 | mean center error mm | mean bbox IoU | mean doctor ROI coverage | mean bone overlap |
|---|---:|---:|---:|---:|
| 老师：单肱骨锚点 | 7.771 | 0.0578 | 0.0851 | 0.0006 |
| 旧多骨 | 6.311 | 0.1089 | 0.1855 | 0.0029 |
| 新多骨：margin + z continuity | 5.973 | 0.1161 | 0.2153 | 0.0022 |
| 旧融合 | 5.954 | 0.1178 | 0.2075 | 0.0020 |
| 新融合 | 5.776 | 0.1248 | 0.2197 | 0.0015 |

结论：骨 margin 与 z 连续性同时改善了中心误差、IoU、coverage，并降低了多骨方法的平均骨重叠。与老师保底融合后，当前最好结果为：

```text
mean center error: 5.776 mm
mean bbox IoU: 0.1248
mean doctor ROI coverage: 0.2197
mean bone overlap: 0.0015
```

## 4. 每例变化

| 病例 | 旧多骨 center error | 新多骨 center error | 旧 coverage | 新 coverage | 旧 bone overlap | 新 bone overlap | 复盘 |
|---|---:|---:|---:|---:|---:|---:|---|
| HMC | 5.81 | 4.53 | 0.0601 | 0.2280 | 0.0000 | 0.0000 | z 连续性和重排明显改善 |
| LHY | 3.72 | 3.75 | 0.1844 | 0.1821 | 0.0016 | 0.0010 | 基本持平，骨重叠降低 |
| LWL | 5.50 | 5.51 | 0.1929 | 0.1787 | 0.0000 | 0.0000 | 小幅损失，可接受 |
| OSQ | 10.95 | 6.03 | 0.0000 | 0.1268 | 0.0037 | 0.0102 | 好候选被提前，但仍贴骨；融合仍回退老师 |
| SB | 10.63 | 10.63 | 0.0000 | 0.0000 | 0.0031 | 0.0031 | 仍失败，问题是候选 z 空间没覆盖医生 ROI |
| WQX | 9.96 | 9.61 | 0.0263 | 0.0251 | 0.0005 | 0.0000 | 中心略好，骨重叠归零 |
| YPL | 4.68 | 4.49 | 0.2801 | 0.2517 | 0.0102 | 0.0063 | margin 生效，骨重叠明显下降 |
| ZH | 2.74 | 2.74 | 0.5740 | 0.5740 | 0.0000 | 0.0000 | 保持强正例 |
| ZJ | 4.95 | 8.27 | 0.2473 | 0.2968 | 0.0086 | 0.0000 | 避骨过强导致中心偏移；融合回退老师 |
| ZJY | 4.17 | 4.17 | 0.2896 | 0.2896 | 0.0010 | 0.0010 | 保持稳定 |

## 5. 当前判断

本轮优化有效，尤其解决了两个原问题：

- YPL、WQX、ZJ 的骨重叠得到明显压制。
- OSQ 的正确方向候选被明显提前，不再完全排不到前面。

但还有两个风险：

1. Margin 过强时可能把 ROI 推离医生中心，例如 ZJ。
2. SB 的医生 ROI 位于更低 z 层，当前候选生成仍没有覆盖，需要下一轮专门扩展 z 搜索与候选生成。

## 6. 下一步建议

优先处理 SB 类失败：

1. 增加低 z 层探索分支，不只依赖当前肱骨-顶板组合出现的层面。
2. 在多骨候选之外加入“老师 z 层附近的多骨微调候选”。
3. 将融合规则从固定 `score < 3.7` 升级为多因素规则：score、margin、continuity、teacher/multibone z 差异共同决策。

