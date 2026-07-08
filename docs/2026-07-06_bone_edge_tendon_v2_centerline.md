# 2026-07-06 骨边缘中心线候选与保守接触复核

## 改动目的

继续优化传统形态学定位方法，验证“先做骨分割，然后沿骨骼边缘找肌腱通道”的路线。重点判断目前效果差的原因是骨分割不准，还是分割后的候选选择逻辑不够好。

## 代码改动

- `src/supraspinatus_locator/localization/multi_bone_traditional.py`
  - 将 `bone_edge_tendon` 从单点 ROI 升级为中心线/带状 ROI。
  - 新增局部 ROI 统计，避免为每个候选生成整幅 3D mask 后过慢。
  - 新增骨边缘候选特征：edge angle、surface normal、surface distance、cortical support、soft tissue band、edge continuity、arc residual 等。
- `scripts/run_multibone_locator.py`
  - 新增 bone edge centerline 相关参数。
- `scripts/run_multibone_dicom_inference.py`
  - 支持无标注 DICOM 数据导出 bone edge 候选。
- `scripts/render_bone_edge_tendon_previews.py`
  - 输出骨边缘 top-k 候选预览。
- `scripts/run_candidate_ranker_experiment.py`
  - 加入 `bone_edge_tendon_only` 对照。
  - 新增保守 `contact_like` rescue。
  - 对 surface/edge 候选改为 review alternative，而不是直接最终接管。

## 实验输出

主目录：

- `outputs/2026-07_bone_edge_tendon_v2_centerline`

关键结果：

- 旧 10 例主流程结果：mean error 4.161 mm，worst 6.040 mm。
- `bone_edge_tendon_only`：mean error 14.114 mm，说明独立接管不稳。
- `unified_ranker_policy + contact review`：mean error 5.467 mm，worst 9.610 mm。
- `loocv_logistic_ranker`：mean error 6.142 mm，worst 9.610 mm。

## 结论

本轮结果支持当前判断：骨分割/骨边缘提取不是最主要瓶颈。`bone_edge_tendon` 已经能在 WQX 等病例中生成 3-4 mm 级候选，但自动排序器仍可能不选它；同时 OSQ/ZJ 说明错误骨边缘也可能满足贴骨和连续条件。因此，后续重点应放在更稳健的候选排序、人工反馈闭环和区分正确上外侧弧段/错误骨边缘的特征上。

## 验证

已执行：

```powershell
python -m py_compile scripts\run_candidate_ranker_experiment.py scripts\render_bone_edge_tendon_previews.py scripts\run_multibone_locator.py scripts\run_multibone_dicom_inference.py src\supraspinatus_locator\localization\multi_bone_traditional.py
python scripts\run_candidate_ranker_experiment.py --labeled-candidates outputs\2026-07_bone_edge_tendon_v2_centerline\labeled\results\per_case_topk.csv --unlabeled-candidates outputs\2026-07_bone_edge_tendon_v2_centerline\unlabeled\results\per_case_topk.csv --unlabeled-feedback outputs\2026-07_bone_edge_tendon_v2_centerline\unlabeled\results\unlabeled_visual_feedback_template.csv --old-best-final outputs\surface_arc_best_final\results\per_case_final.csv --generalized-final outputs\2026-07-02_labeled_generalized_rescue_no_teacher\results\per_case_final.csv --output-dir outputs\2026-07_bone_edge_tendon_v2_centerline\ranker_contact_review
python scripts\render_bone_edge_tendon_previews.py --labeled-candidates outputs\2026-07_bone_edge_tendon_v2_centerline\labeled\results\per_case_topk.csv --unlabeled-candidates outputs\2026-07_bone_edge_tendon_v2_centerline\unlabeled\results\per_case_topk.csv --output-dir outputs\2026-07_bone_edge_tendon_v2_centerline\previews --topk 8
python scripts\render_current_method_previews.py --labeled-selections outputs\2026-07_bone_edge_tendon_v2_centerline\ranker_contact_review\results\unified_learned_labeled_selections.csv --unlabeled-selections outputs\2026-07_bone_edge_tendon_v2_centerline\ranker_contact_review\results\unlabeled_unified_selections.csv --output-dir outputs\2026-07_bone_edge_tendon_v2_centerline\ranker_contact_review_previews
```

