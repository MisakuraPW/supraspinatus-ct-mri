# 2026-07-06 Bone Edge Tendon Experiment

## 目标

尝试一套新的传统形态学路线：先做骨分割，提取肱骨头外上侧骨皮质边缘，再沿骨表面外法线寻找疑似冈上肌腱 ROI。

这条路线作为新候选源 `bone_edge_tendon` 接入现有候选池，不替换当前多骨/弧面/排序器主流程。

## 代码变更

- `src/supraspinatus_locator/localization/multi_bone_traditional.py`
  - 新增 `bone_edge_tendon` 候选生成器。
  - 新增骨边缘局部圆拟合、2D 腐蚀边缘提取、软组织带统计。
  - 新增边缘特征字段：`edge_angle_deg`、`edge_point_x/y/z`、`surface_normal_x/y`、`surface_distance_mm`、`cortical_edge_support`、`soft_tissue_band_mean/std`、`arc_fit_residual` 等。

- `scripts/run_multibone_locator.py`
  - 新增 `--bone-edge-enable` 及相关参数。

- `scripts/run_multibone_dicom_inference.py`
  - DICOM 无标注推理支持 `bone_edge_tendon` 候选导出。

- `scripts/run_candidate_ranker_experiment.py`
  - 新增 `bone_edge_tendon` source feature。
  - 新增 `bone_edge_tendon_only` 独立评估策略。

- `scripts/render_bone_edge_tendon_previews.py`
  - 新增骨边缘候选专门预览脚本。

## 实验输出

- 候选池：
  - `outputs/2026-07_bone_edge_tendon_probe/labeled`
  - `outputs/2026-07_bone_edge_tendon_probe/unlabeled`

- 排序实验：
  - `outputs/2026-07_bone_edge_tendon_probe/ranker`

- 骨边缘候选预览：
  - `outputs/2026-07_bone_edge_tendon_probe/previews`

- 接入骨边缘后的最终预览：
  - `outputs/2026-07_bone_edge_tendon_probe/ranker_previews`

- 报告：
  - `outputs/2026-07_bone_edge_tendon_probe/reports/bone_edge_tendon_experiment_report.md`

## 关键结果

| 方法 | 平均误差 mm | 最差误差 mm | coverage | IoU | bone overlap |
|---|---:|---:|---:|---:|---:|
| old_best_policy | 3.882 | 6.04 | 0.3880 | 0.2164 | 0.0161 |
| previous unified_ranker_policy | 5.259 | 9.61 | 0.2582 | 0.1506 | 0.0105 |
| bone_edge_tendon_only | 14.392 | 28.05 | 0.0528 | 0.0213 | 0.0274 |
| unified_ranker_with_bone_edge | 5.765 | 9.61 | 0.2141 | 0.1222 | 0.0180 |
| LOOCV with bone_edge | 6.543 | 14.16 | 0.2129 | 0.1195 | 0.0169 |

## 结论

第一版 `bone_edge_tendon` 不适合独立作为最终定位方法。它大多能找到肱骨头外上侧骨皮质边缘，但 ROI 偏向骨顶外侧小框，没有充分覆盖肌腱沿肱骨头包绕的带状区域。

它仍可作为候选源保留，因为 WQX、YPL 等病例中存在较好的骨边缘候选。下一步更应该从“单点法线框”升级为“沿骨边缘的弧形中心线/带状 ROI”。

## 复现实验命令

旧 10 例：

```powershell
python scripts\run_multibone_locator.py --data-dir outputs\normalized_10cases --output-dir outputs\2026-07_bone_edge_tendon_probe\labeled --teacher-csv outputs\teacher_10cases\evaluation\ct_tendon_locator_results.csv --bone-margin-voxels 2 --low-z-enable --contact-z-enable --contact-z-select-enable --teacher-z-refine-enable --surface-arc-enable --surface-arc-select-enable --bone-edge-enable --bone-edge-anchor-count 8 --topk 8 --export-candidates --candidate-preview-topk 8
```

新 10 例：

```powershell
python scripts\run_multibone_dicom_inference.py --data-dir Data\260626-ct-10例 --output-dir outputs\2026-07_bone_edge_tendon_probe\unlabeled --selection-policy generalized --bone-margin-voxels 2 --low-z-enable --surface-arc-enable --surface-arc-select-enable --bone-edge-enable --bone-edge-anchor-count 8 --topk 8 --export-candidates --candidate-preview-topk 8
```

排序实验：

```powershell
python scripts\run_candidate_ranker_experiment.py --labeled-candidates outputs\2026-07_bone_edge_tendon_probe\labeled\results\per_case_topk.csv --unlabeled-candidates outputs\2026-07_bone_edge_tendon_probe\unlabeled\results\per_case_topk.csv --unlabeled-feedback outputs\2026-07_bone_edge_tendon_probe\unlabeled\results\unlabeled_visual_feedback_template.csv --old-best-final outputs\surface_arc_best_final\results\per_case_final.csv --generalized-final outputs\2026-07-02_labeled_generalized_rescue_no_teacher\results\per_case_final.csv --output-dir outputs\2026-07_bone_edge_tendon_probe\ranker
```
