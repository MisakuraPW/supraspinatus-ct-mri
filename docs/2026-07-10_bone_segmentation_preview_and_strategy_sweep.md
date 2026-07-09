# 2026-07-10 骨分割预览与 TotalSeg-guided 定位策略 sweep

## 目的

本轮新增两类实验代码：

1. 骨分割对比预览：肉眼比较传统 HU 阈值骨分割、TotalSegmentator 深度学习骨分割、TotalSeg-guided HU 骨分割。
2. 定位策略 sweep：在“TotalSeg 深度学习骨分割 + 传统形态学定位”的框架下，一次性跑多组选择策略，判断到底是候选生成问题还是最终选择器问题。

## 先生成 TotalSegmentator 肩部骨 mask

如果云端还没有生成 TotalSegmentator 骨 mask，先运行：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --fast \
  --device gpu \
  --require-gpu \
  --skip-existing \
  --stop-on-fail
```

如果某些病例 `combined=0` 或明显过小，不要立刻删掉；后续预览图可以帮助判断它为什么不可靠。

## 生成骨分割对比预览图

```bash
python scripts/preview_bone_segmentation_compare.py \
  --data-dir Data/label \
  --totalseg-mask-dir outputs/2026-07_totalseg_shoulder_bones \
  --output-dir outputs/2026-07_bone_segmentation_compare \
  --guided-dilation-voxels 4
```

输出：

```text
outputs/2026-07_bone_segmentation_compare/previews/*_bone_compare.png
outputs/2026-07_bone_segmentation_compare/bone_segmentation_compare_summary.csv
```

每张图包含四列：

1. 原始 CT。
2. `HU > 300` 传统形态学骨 mask。
3. TotalSegmentator 肩部骨 mask。
4. TotalSeg-guided HU mask。

重点看：

- TotalSeg 是否漏掉肱骨头、肩胛骨、锁骨。
- TotalSeg 是否只分到很小一块。
- TotalSeg-guided HU 是否保留了骨皮质边缘。
- 传统 HU 是否包含过多无关骨结构。

## 一次性跑多组定位策略

核心命令：

```bash
python scripts/run_totalseg_locator_strategy_sweep.py \
  --data-dir Data/label \
  --bone-mask-dir outputs/2026-07_totalseg_shoulder_bones \
  --output-root outputs/2026-07_totalseg_locator_strategy_sweep
```

如果只想先快速验证核心策略：

```bash
python scripts/run_totalseg_locator_strategy_sweep.py \
  --data-dir Data/label \
  --bone-mask-dir outputs/2026-07_totalseg_shoulder_bones \
  --output-root outputs/2026-07_totalseg_locator_strategy_sweep_quick \
  --quick
```

如果只想跑传统选择器，不跑候选级学习排序器，可加：

```bash
--skip-ranker
```

## sweep 中包含的策略

默认完整 sweep 包含：

- `threshold_roi_generalized`：上一轮 generalized 对照。
- `threshold_roi_best_score`：直接选原始总分最高候选。
- `threshold_roi_current_first`：优先旧多骨锚点候选。
- `threshold_roi_conservative`：重新加权，降低 surface_arc 无条件优先级。
- `threshold_roi_consensus`：奖励不同候选源在空间上达成一致。
- `threshold_roi_edge_priority`：提高 bone_edge_tendon 候选权重。
- `threshold_roi_surface_suppressed`：强力抑制孤立 surface_arc 候选。
- `hu_conservative_baseline`：不用 TotalSeg，仅 HU 阈值 + 新保守选择器。
- `threshold_roi_oracle_upper_bound`：使用医生 ROI 选 top-k 中最佳候选，只作为候选池理论上限，不能当真实方法。
- `conservative_dilate2/dilate8`：测试 TotalSeg-guided HU ROI 膨胀半径影响。
- `direct_conservative_probe`：直接使用 TotalSeg mask 的探针实验，通常不推荐作为最终方案。

每组实验还会自动运行一次候选级 ranker：

- `unified_static_policy`：统一静态打分。
- `unified_ranker_policy`：用旧 10 例候选标签训练的轻量排序器。
- `loocv_ranker_policy`：leave-one-case-out 验证，重点看泛化风险。
- `pairwise_ranker_policy`：候选两两比较训练。
- `loocv_pairwise_ranker`：pairwise 的 leave-one-case-out 验证。

这些 ranker 结果可以判断“候选池本身有救，但手写选择器不稳”这个假设是否成立。

注意：`unified_ranker_policy` 和 `pairwise_ranker_policy` 是训练集内结果，可能偏乐观；真正判断泛用性应优先看 `loocv_logistic_ranker` 和 `loocv_pairwise_ranker`。

## 下载给我分析的重点结果

优先下载整个目录：

```text
outputs/2026-07_bone_segmentation_compare
outputs/2026-07_totalseg_locator_strategy_sweep
```

如果文件太大，至少下载：

```text
outputs/2026-07_bone_segmentation_compare/bone_segmentation_compare_summary.csv
outputs/2026-07_bone_segmentation_compare/previews
outputs/2026-07_totalseg_locator_strategy_sweep/ranked_summary_metrics.csv
outputs/2026-07_totalseg_locator_strategy_sweep/combined_summary_metrics.csv
outputs/2026-07_totalseg_locator_strategy_sweep/combined_ranker_policy_summary.csv
outputs/2026-07_totalseg_locator_strategy_sweep/combined_per_case_final.csv
outputs/2026-07_totalseg_locator_strategy_sweep/*/results/failure_analysis.csv
outputs/2026-07_totalseg_locator_strategy_sweep/*/ranker/results/policy_summary.csv
outputs/2026-07_totalseg_locator_strategy_sweep/*/previews
```

## 解释

如果 `oracle_upper_bound` 接近旧最优，而真实选择器很差，说明候选池有潜力，主要是排序器问题。

如果 `oracle_upper_bound` 也很差，说明 TotalSeg-guided 候选生成本身没有提供足够好的候选，应回退到旧最优方法或重新设计候选生成。
