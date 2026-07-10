# 2026-07-10 TotalSeg sweep 结果分析与下一步

## 这轮结果结论

骨分割对比显示：

- `LHY`、`ZH` 的 TotalSeg 肩部骨 mask 为空。
- `HMC` 的 TotalSeg mask 只有 `4608` 体素，明显过小。
- 其余病例 TotalSeg 可以提供空间先验，但它是实心语义骨，不能直接替代 HU 骨皮质。

定位策略 sweep 显示：

- `oracle` 平均误差约 `4.247mm`，说明候选池里仍有好候选。
- 最好的真实手工策略是 `current_first`，平均误差约 `8.202mm`，仍明显差于旧最优 `3.882mm`。
- `generalized` 和直接 `best_score` 容易被 `surface_arc` 带偏。
- 训练集 ranker 可达到约 `4.965mm`，但 LOOCV 明显退化，说明学习排序器在 10 例上仍有过拟合风险。

## 方法判断

TotalSeg 骨分割目前不适合作为最终主干替代旧 HU 形态学骨皮质。更合理的定位是：

1. TotalSeg 作为可选空间先验；
2. TotalSeg 异常时自动回退 HU；
3. 最终选择器以 `current_multibone` 为保守主干；
4. `bone_edge_tendon` 只在 current 明显弱时作为救援；
5. `surface_arc` 不再无条件接管；
6. 对候选源冲突大的病例输出 review 信息。

## 本次代码更新

新增定位策略：

```text
adaptive_edge_guarded
```

规则概述：

- 默认选择 `current_multibone`。
- 如果 TotalSeg mask 为空或过小，直接使用 current，避免错误深度学习骨后端影响定位。
- 当 current 候选连续性低、锚点支持少、或明显脱骨时，才允许 `bone_edge_tendon` 救援。
- 救援候选必须满足位置、骨重叠、近骨比例、半径归一化距离等质控。
- 如果 current 与 edge 候选分歧较大，会写入 `needs_review` 和 `review_alternative_*` 字段。

基于已下载候选池的离线验证，`adaptive_edge_guarded` 约为：

| 指标 | 数值 |
|---|---:|
| 平均误差 | 5.565 mm |
| 最差误差 | 10.69 mm |
| 平均覆盖率 | 0.2256 |

这比 `current_first` 更稳，但仍弱于旧最优 `surface_arc_best_final`。因此它应作为下一轮重点实验策略，而不是直接替换最终方案。

新增结果分析脚本：

```bash
python scripts/analyze_totalseg_sweep_results.py \
  --bone-compare-dir outputs/2026-07_bone_segmentation_compare \
  --sweep-dir outputs/2026-07_totalseg_locator_strategy_sweep
```

默认生成：

```text
outputs/2026-07_totalseg_locator_strategy_sweep/reports/sweep_analysis_report.md
```

## 下一轮云端运行

建议先跑 quick：

```bash
python scripts/run_totalseg_locator_strategy_sweep.py \
  --data-dir Data/label \
  --bone-mask-dir outputs/2026-07_totalseg_shoulder_bones \
  --output-root outputs/2026-07_totalseg_locator_strategy_sweep_v2 \
  --quick
```

然后生成自动分析报告：

```bash
python scripts/analyze_totalseg_sweep_results.py \
  --bone-compare-dir outputs/2026-07_bone_segmentation_compare \
  --sweep-dir outputs/2026-07_totalseg_locator_strategy_sweep_v2
```

如果 quick 里 `adaptive_edge_guarded` 优于 `current_first`，再跑完整 sweep。
