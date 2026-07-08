# 2026-07-02 泛化策略回测原有标注 10 例

## 目的

将无标注 DICOM 上调整出的 `generalized + cross-anchor rescue` 策略，重新落回原有带医生 ROI 标注的 10 例数据，评估其对原标注指标的影响。

用户目标：允许性能有一定下降，但优先希望方法更泛用。

## 运行配置

输出目录：

`outputs/2026-07-02_labeled_generalized_rescue_no_teacher`

运行命令核心：

```powershell
python scripts\run_multibone_locator.py `
  --data-dir outputs\normalized_10cases `
  --output-dir outputs\2026-07-02_labeled_generalized_rescue_no_teacher `
  --selection-policy generalized `
  --current-anchor-count 160 `
  --bone-margin-voxels 2 `
  --continuity-window 2 `
  --continuity-xy-tolerance 14 `
  --low-z-enable `
  --branch-anchor-count 6 `
  --surface-arc-enable `
  --surface-arc-select-enable `
  --surface-arc-sphere-blend 0 `
  --topk 5
```

说明：这次按无标注 DICOM 上的“当前泛化方法”回测，未启用旧实验中的 `contact_z` 和 `teacher_z_refine` 分支。

## 总体结果

| 方法 | mean center error | worst error | mean bbox IoU | mean doctor ROI coverage | mean bone overlap |
|---|---:|---:|---:|---:|---:|
| 原标注十例最优版 `surface_arc_best_final` | 3.882 | 6.040 | 0.2164 | 0.3880 | 0.0161 |
| 当前泛化回测版 | 12.172 | 30.580 | 0.0435 | 0.1025 | 0.0400 |

## 逐例结果

| 病例 | 原方法 | 新方法 | 原误差 | 新误差 | 原 coverage | 新 coverage | 原骨重叠 | 新骨重叠 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| HMC | current_multibone | surface_arc | 4.53 | 10.92 | 0.2280 | 0.2228 | 0.0000 | 0.0261 |
| LHY | current_multibone | surface_arc | 5.19 | 12.97 | 0.1535 | 0.0871 | 0.0000 | 0.0557 |
| LWL | surface_arc | surface_arc | 0.50 | 6.24 | 0.6567 | 0.1972 | 0.0565 | 0.0450 |
| OSQ | current_multibone | surface_arc | 6.04 | 14.76 | 0.3647 | 0.0000 | 0.0000 | 0.0384 |
| SB | contact_z | surface_arc | 1.90 | 8.29 | 0.6846 | 0.0000 | 0.0510 | 0.0444 |
| WQX | surface_arc | surface_arc | 4.40 | 7.03 | 0.2593 | 0.1159 | 0.0329 | 0.0329 |
| YPL | current_multibone | surface_arc | 4.52 | 5.73 | 0.2666 | 0.1448 | 0.0081 | 0.0400 |
| ZH | current_multibone | surface_arc | 2.74 | 30.58 | 0.5740 | 0.0000 | 0.0000 | 0.0356 |
| ZJ | teacher_baseline | surface_arc | 4.27 | 13.55 | 0.4297 | 0.0000 | 0.0127 | 0.0418 |
| ZJY | current_multibone | surface_arc | 4.73 | 11.65 | 0.2625 | 0.2574 | 0.0000 | 0.0400 |

## 结论

当前 `generalized + rescue` 策略在原有标注 10 例上退化过大，不能直接作为正式替代方案。

它的优点是更倾向于贴近肱骨头弧面，视觉上更符合“肌腱弧形包绕肱骨头”的理解；但它与原有医生 ROI 标注之间存在明显冲突，尤其在 ZH、OSQ、SB、ZJ 等病例上 coverage 直接为 0。

这说明现在不是简单地“用 generalized 替换旧方法”的阶段。更稳妥的方向是：

1. 保留 `surface_arc` 作为候选生成能力，而不是无条件大面积接管。
2. 用原 10 例标注和新增无标注视觉质控共同设计候选选择器。
3. 对低骨重叠 current、贴骨 surface_arc、contact_z 等候选同时输出 top-k，先做人工复核。
4. 后续若医生认可“更贴肱骨头”的定义，需要重新补标或调整标注标准，否则定量指标会天然惩罚这类候选。

## 产物

- 指标：`outputs/2026-07-02_labeled_generalized_rescue_no_teacher/results/summary_metrics.csv`
- 逐例结果：`outputs/2026-07-02_labeled_generalized_rescue_no_teacher/results/per_case_final.csv`
- 预览拼图：`outputs/2026-07-02_labeled_generalized_rescue_no_teacher/labeled_10cases_generalized_rescue_preview_montage.png`
