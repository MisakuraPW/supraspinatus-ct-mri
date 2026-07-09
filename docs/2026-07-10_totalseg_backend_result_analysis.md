# 2026-07-10 TotalSeg 骨后端定位结果分析

## 输出位置

本次分析目录：

```text
outputs/outputs/2026-07_totalseg_bone_backend_locator
```

本地没有 `outputs/outpus`，实际结果在 `outputs/outputs` 中。

## 总体结果

本轮使用 TotalSegmentator 肩部骨 mask 作为骨后端，并启用：

```text
--external-bone-mode threshold_roi
--allow-threshold-bone-fallback
--surface-arc-enable
--bone-edge-enable
--selection-policy generalized
```

最终定量指标：

| 方法 | 平均误差 mm | 中位误差 mm | 最差误差 mm | 平均覆盖率 | 平均 IoU | 骨重叠 |
|---|---:|---:|---:|---:|---:|---:|
| 旧最优 surface_arc_best_final | 3.882 | 4.460 | 6.040 | 0.3880 | 0.2164 | 0.0161 |
| 本轮 TotalSeg-guided backend | 12.433 | 11.685 | 30.670 | 0.1333 | 0.0563 | 0.0382 |

结论：本轮不是改进，而是明显退化。

## 逐例结果

| 病例 | 本轮误差 mm | 覆盖率 | IoU | 骨重叠 | 主要问题 |
|---|---:|---:|---:|---:|---|
| HMC | 8.94 | 0.2332 | 0.1124 | 0.0212 | TotalSeg mask 过小，回退 HU；最终仍有 XY 偏差 |
| LHY | 12.93 | 0.0871 | 0.0473 | 0.0557 | TotalSeg mask 为空，回退 HU；选择器选错候选 |
| LWL | 6.24 | 0.2014 | 0.0792 | 0.0361 | threshold_roi 可用，但覆盖率下降 |
| OSQ | 15.50 | 0.0000 | 0.0000 | 0.0329 | surface_arc 候选排前，但医生 ROI 覆盖为 0 |
| SB | 11.97 | 0.0000 | 0.0000 | 0.0476 | surface_arc 排序过强，错过更好候选 |
| WQX | 4.27 | 0.2748 | 0.1224 | 0.0392 | 本轮与旧方法接近，属于少数稳定例 |
| YPL | 6.35 | 0.2788 | 0.1054 | 0.0416 | 小幅退化 |
| ZH | 30.67 | 0.0000 | 0.0000 | 0.0314 | 最严重，候选池有好候选但最终选错 |
| ZJ | 16.06 | 0.0000 | 0.0000 | 0.0361 | surface_arc 选错弧段/层面 |
| ZJY | 11.40 | 0.2574 | 0.0966 | 0.0400 | 有可用候选，但最终不佳 |

## TotalSeg 后端是否失败

不是所有病例都失败，但 TotalSeg 作为“直接骨 mask 替换”并不适合当前旧锚点逻辑。

本轮最终病例中：

- `threshold_roi` 使用 7 例。
- TotalSeg mask 过小或为空而回退 HU 使用 3 例：HMC、LHY、ZH。
- 最终选择源全部为 `surface_arc`，没有选择 `current_multibone` 或 `bone_edge_tendon`。

这说明 TotalSeg 的空间先验本身不是完全没用，但当前选择器对 `surface_arc` 的偏好过强。

## 最关键发现

候选池本身仍然有潜力：

| 指标 | 最终选择 | 候选池 oracle |
|---|---:|---:|
| 平均误差 mm | 12.433 | 4.247 |
| 最差误差 mm | 30.670 | 7.700 |
| 平均覆盖率 | 0.1333 | 0.3446 |

`oracle` 指的是：如果每例都能从 top-k 候选中选出最接近医生 ROI 的候选。

因此当前主要瓶颈不是候选生成，而是排序/选择策略。尤其是 `ZH`：

- 最终选择：surface_arc，误差 30.67mm，覆盖率 0。
- top-k 中存在 current_multibone 候选，误差 2.07mm，覆盖率 0.6055。

这类病例说明“好候选已经生成了，但没有被选中”。

## 判断

本轮 TotalSegmentator 骨后端不应直接进入最终方案。它可以保留为候选生成/空间先验实验，但不能替代旧最优方法。

下一步优先方向不是继续调 TotalSeg mask，而是修复统一选择器：

1. 降低 `surface_arc` 的无条件优先级。
2. 对 coverage proxy、z 层一致性、候选源分歧加入更强质控。
3. 当 `surface_arc` 与 `current_multibone` 在同例中差异很大时，输出 top-3 复核或选择更保守候选。
4. 用旧 10 例医生 ROI 做候选级 leave-one-case-out 排序器，而不是继续手写 surface_arc 接管规则。

