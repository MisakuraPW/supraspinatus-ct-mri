# 2026-07-06 Anchor Review 与 Pairwise Ranker 实验记录

## 改动目的

继续优化骨边缘中心线方法后的候选选择问题。上一轮结论是：骨分割和骨边缘候选生成不是主要瓶颈，主要瓶颈在于如何从多源候选池中选择最终 ROI。

## 代码改动

- `scripts/run_candidate_ranker_experiment.py`
  - 新增 `pairwise_ranker_policy` 和 `loocv_pairwise_ranker`。
  - 新增 `anchor_review_policy`。
  - 新增参数：
    - `--labeled-anchor-final`
    - `--unlabeled-anchor-final`
  - 新增输出：
    - `anchor_labeled_selections.csv`
    - `unlabeled_anchor_selections.csv`
    - `pairwise_labeled_selections.csv`
    - `pairwise_loocv_predictions.csv`
    - `unlabeled_pairwise_top3.csv`

## 实验结果

输出目录：

- `outputs/2026-07_bone_edge_tendon_v2_anchor_review`

关键指标：

- `anchor_review_policy`: mean error 4.161 mm，worst 6.040 mm，coverage 0.3448。
- `pairwise_ranker_policy`: mean error 6.146 mm，worst 9.610 mm。
- `loocv_pairwise_ranker`: mean error 7.001 mm，worst 15.580 mm。

## 结论

pairwise ranker 是负结果：旧 10 例不足以支撑更复杂的排序器，留一验证变差。当前更稳的策略是使用主流程最终框作为 anchor，然后通过候选池提供 review alternative，而不是让不稳定的新候选直接接管最终结果。

## 验证命令

```powershell
python -m py_compile scripts\run_candidate_ranker_experiment.py
python scripts\run_candidate_ranker_experiment.py --labeled-candidates outputs\2026-07_bone_edge_tendon_v2_centerline\labeled\results\per_case_topk.csv --unlabeled-candidates outputs\2026-07_bone_edge_tendon_v2_centerline\unlabeled\results\per_case_topk.csv --unlabeled-feedback outputs\2026-07_bone_edge_tendon_v2_centerline\unlabeled\results\unlabeled_visual_feedback_template.csv --labeled-anchor-final outputs\2026-07_bone_edge_tendon_v2_centerline\labeled\results\per_case_final.csv --unlabeled-anchor-final outputs\2026-07_bone_edge_tendon_v2_centerline\unlabeled\results\per_case_inference.csv --old-best-final outputs\surface_arc_best_final\results\per_case_final.csv --generalized-final outputs\2026-07-02_labeled_generalized_rescue_no_teacher\results\per_case_final.csv --output-dir outputs\2026-07_bone_edge_tendon_v2_anchor_review
python scripts\render_current_method_previews.py --labeled-selections outputs\2026-07_bone_edge_tendon_v2_anchor_review\results\anchor_labeled_selections.csv --unlabeled-selections outputs\2026-07_bone_edge_tendon_v2_anchor_review\results\unlabeled_anchor_selections.csv --output-dir outputs\2026-07_bone_edge_tendon_v2_anchor_review\previews
```

