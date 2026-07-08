# 2026-07-06 肱骨头锚点识别修复

## 改动目的

修复无标注第 5 例 `杨国玲` 定位明显跑到肩胛/盂侧的问题。复盘确认该问题不是肱骨头没有分割出来，而是骨组件身份识别错误：后续 surface/bone-edge 候选依赖错误 anchor，导致正确 bone-edge 候选缺失。

## 代码改动

- `src/supraspinatus_locator/localization/multi_bone_traditional.py`
  - 新增 `humeral_head_likeness()`。
  - 将 `head_prior` 加入 anchor score。
  - 将 `head_prior` 加入 current/surface/bone-edge 候选分数。
  - 收紧 surface_arc 自动接管条件：
    - `z_continuity_score >= 0.75`
    - `near_bone_fraction >= 0.045`
    - `center_y_minus_humerus_top <= 9.5`
  - 将 teacher 回退限制为低分 current anchor 且 `possible_wrong_bone=True` 的场景，避免 SB/WQX 被错误回退。

## 实验输出

主目录：

- `outputs/2026-07_humeral_head_anchor_fix_final`

旧 10 例：

- mean error: `4.360 mm`
- worst error: `8.270 mm`
- coverage: `0.3445`
- bone overlap: `0.0155`

无标注 10 例：

- `杨国玲` 已从错误肩胛侧回到肱骨头上方。
- 每例均有 `bone_edge_tendon` top-k 候选。

## 验证命令

```powershell
python -m py_compile src\supraspinatus_locator\localization\multi_bone_traditional.py
python scripts\run_multibone_locator.py --data-dir outputs\normalized_10cases --output-dir outputs\2026-07_humeral_head_anchor_fix_final\labeled --teacher-csv outputs\teacher_10cases\evaluation\ct_tendon_locator_results.csv --bone-margin-voxels 2 --low-z-enable --contact-z-enable --contact-z-select-enable --teacher-z-refine-enable --surface-arc-enable --surface-arc-select-enable --bone-edge-enable --bone-edge-anchor-count 8 --bone-edge-z-window 0 --bone-edge-centerline-enable --bone-edge-centerline-points 5 --bone-edge-centerline-angle-step-deg 8 --bone-edge-centerline-half-size 12 5 1 --bone-edge-channel-downshift-voxels 6 --topk 8 --export-candidates --candidate-preview-topk 8
python scripts\run_multibone_dicom_inference.py --data-dir Data\260626-ct-10例 --output-dir outputs\2026-07_humeral_head_anchor_fix_final\unlabeled --selection-policy generalized --bone-margin-voxels 2 --low-z-enable --surface-arc-enable --surface-arc-select-enable --bone-edge-enable --bone-edge-anchor-count 8 --bone-edge-z-window 0 --bone-edge-centerline-enable --bone-edge-centerline-points 5 --bone-edge-centerline-angle-step-deg 8 --bone-edge-centerline-half-size 12 5 1 --bone-edge-channel-downshift-voxels 6 --topk 8 --export-candidates --candidate-preview-topk 8
python scripts\render_current_method_previews.py --labeled-selections outputs\2026-07_humeral_head_anchor_fix_final\labeled\results\per_case_final.csv --unlabeled-selections outputs\2026-07_humeral_head_anchor_fix_final\unlabeled\results\per_case_inference.csv --output-dir outputs\2026-07_humeral_head_anchor_fix_final\previews
python scripts\render_bone_edge_tendon_previews.py --labeled-candidates outputs\2026-07_humeral_head_anchor_fix_final\labeled\results\per_case_topk.csv --unlabeled-candidates outputs\2026-07_humeral_head_anchor_fix_final\unlabeled\results\per_case_topk.csv --output-dir outputs\2026-07_humeral_head_anchor_fix_final\bone_edge_previews --topk 8
```

