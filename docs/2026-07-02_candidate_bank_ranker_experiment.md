# 2026-07-02 Candidate Bank + Robust Ranker Experiment

## 本轮目标

本轮按“传统形态学定位方法的稳健化计划”实现候选池化与统一排序，不再让 `current_multibone`、`surface_arc`、`low_z`、`contact_z` 等路线各自抢最终答案。

核心变化：

- 定位脚本新增候选导出模式：每例输出统一候选 CSV 与 top-k 预览图。
- 无标注 DICOM 推理新增人工视觉反馈模板。
- 新增统一排序实验脚本，比较旧最优、泛化策略、统一静态排序器、轻量学习排序器、LOOCV。
- 排序器新增两类统一复核规则：`contact_like_anatomic_review` 与 `surface_margin_too_deep_review`，都基于形态特征触发，不按病例名触发。

## 代码变更

- `src/supraspinatus_locator/localization/multi_bone_traditional.py`
  - 补齐候选特征字段：弧角、表面偏移、半径归一化距离、同锚点支持数、锚点宽高、连续性、质控 flags。
  - 新增 `save_candidate_sheet_pil`，用于每例候选 top-k 预览。
  - `process_dataset` 新增 `export_candidates` 与 `candidate_preview_topk`。

- `scripts/run_multibone_locator.py`
  - 新增 `--export-candidates` 与 `--candidate-preview-topk`。

- `scripts/run_multibone_dicom_inference.py`
  - 新增无标注候选池导出。
  - 新增 `unlabeled_visual_feedback_template.csv`。
  - DICOM 候选按来源保留 top-k，避免单一路线淹没候选池。

- `scripts/run_candidate_ranker_experiment.py`
  - 新增统一候选排序实验。
  - 支持原 10 例自动标签、新 10 例人工视觉反馈标签。
  - 支持 leave-one-case-out 交叉验证。

## 本轮输出

- 旧 10 例候选池：
  - `outputs/2026-07_candidate_bank/labeled/results/per_case_topk.csv`
  - `outputs/2026-07_candidate_bank/labeled/previews/*_candidate_top8.png`

- 新 10 例候选池：
  - `outputs/2026-07_candidate_bank/unlabeled/results/per_case_topk.csv`
  - `outputs/2026-07_candidate_bank/unlabeled/results/unlabeled_visual_feedback_template.csv`
  - `outputs/2026-07_candidate_bank/unlabeled/results/unlabeled_visual_feedback_initial_by_codex.csv`
  - `outputs/2026-07_candidate_bank/unlabeled/previews/*_candidate_top8.png`

- 排序实验：
  - `outputs/2026-07_candidate_ranker_experiment/results/policy_summary.csv`
  - `outputs/2026-07_candidate_ranker_experiment/results/loocv_predictions.csv`
  - `outputs/2026-07_candidate_ranker_experiment/results/unlabeled_unified_top3.csv`

- 带 Codex 初筛反馈的无标注排序实验：
  - `outputs/2026-07_candidate_ranker_experiment_with_visual_feedback/results/unlabeled_unified_selections.csv`

## 关键结果

| policy | mean error | worst error | coverage | IoU | bone overlap |
|---|---:|---:|---:|---:|---:|
| old_best_policy | 3.882 | 6.04 | 0.3880 | 0.2164 | 0.0161 |
| generalized_policy | 12.172 | 30.58 | 0.1025 | 0.0435 | 0.0400 |
| unified_static_policy | 11.991 | 36.39 | 0.2196 | 0.0970 | 0.0344 |
| unified_ranker_policy | 5.259 | 9.61 | 0.2582 | 0.1506 | 0.0105 |
| loocv_logistic_ranker | 5.514 | 9.61 | 0.2684 | 0.1508 | 0.0061 |

解释：

- 旧最优仍是原 10 例定量上限。
- `generalized_policy` 证明“让 surface_arc 全面接管”会在旧标注集大崩。
- `unified_static_policy` 单独不够可靠，说明只靠手写统一打分仍容易被 surface_arc 高分牵走。
- `unified_ranker_policy` 在训练内达到可接受折中。
- `loocv_logistic_ranker` 均值 5.514mm、最差 9.61mm，说明统一候选池 + 轻量排序器已经显著优于 generalized 崩坏版本，但仍弱于旧最优。

## 无标注集视觉反馈

空模板保留在：

`outputs/2026-07_candidate_bank/unlabeled/results/unlabeled_visual_feedback_template.csv`

我额外生成了低置信度初筛：

`outputs/2026-07_candidate_bank/unlabeled/results/unlabeled_visual_feedback_initial_by_codex.csv`

标签分布：

- `acceptable`: 24
- `uncertain`: 22
- `too_high_outer`: 4

注意：这份初筛不是医生标注，只用于测试反馈闭环是否能跑通。

## 当前判断

本轮最重要的进展不是指标超过旧最优，而是把方法结构从“继续堆分支”改成了可复用的候选池和排序框架。现在旧 10 例不会像 generalized 那样崩，新 10 例也能保留 top-3 复核，而不是强行输出单一真值。

后续最高效的路线：

1. 你或医生在 `unlabeled_visual_feedback_template.csv` 里给 top-5 候选打 `good/acceptable/too_high_outer/wrong_bone/too_medial_inner/uncertain`。
2. 重新运行 `scripts/run_candidate_ranker_experiment.py`。
3. 如果有新病例，先导出候选池和预览，再并入反馈，而不是继续写病例分支。
