# 无标注 10 例 fused bone 后端推理入口

本次新增 `scripts/run_multibone_nifti_inference.py`，用于对无标注病例运行定位推理。

## 背景

原有 `run_multibone_dicom_inference.py` 直接读取 `Data/unlabel` 的 DICOM，并使用 HU 阈值骨后端。它不能接收 TotalSeg/HU fused bone mask。

为了让新 10 例也能使用这一轮的骨分割优化，需要先把 DICOM 转为项目标准 NIfTI，再跑 TotalSeg + HU 融合，最后用新增的 NIfTI 推理脚本读取融合骨 mask。

## 推荐流程

```bash
python scripts/prepare_unlabeled_ct_nifti.py \
  --data-dir Data/unlabel \
  --output-dir outputs/2026-07_unlabel_ct_nifti

python scripts/run_totalseg_bone_segmentation_suite.py \
  --data-dir outputs/2026-07_unlabel_ct_nifti \
  --output-root outputs/2026-07_unlabel_totalseg_bone_segmentation_suite \
  --device gpu \
  --require-gpu \
  --skip-existing \
  --totalseg-home-dir /mnt/workspace/supraspinatus-ct-mri/outputs/2026-07_totalseg_weights_cache/.totalsegmentator

python scripts/run_multibone_nifti_inference.py \
  --data-dir outputs/2026-07_unlabel_ct_nifti \
  --output-dir outputs/2026-07_unlabel_fused_locator_inference \
  --bone-mask-dir outputs/2026-07_unlabel_totalseg_bone_segmentation_suite/totalseg_hu_fused \
  --bone-mask-filename shoulder_bones_fused_hu.nii.gz \
  --allow-threshold-bone-fallback \
  --surface-arc-enable \
  --bone-edge-enable \
  --selection-policy adaptive_edge_guarded \
  --export-candidates \
  --candidate-preview-topk 8
```

## 输出

- `results/per_case_inference.csv`：每例最终 ROI 和候选特征。
- `results/per_case_topk.csv`：候选池 top-k。
- `results/unlabeled_visual_feedback_template.csv`：人工视觉反馈模板。
- `previews/*_preview.png`：最终 ROI 预览。
- `previews/*_candidate_top8.png`：候选 top8 预览。

无标注数据没有医生 ROI，因此不会输出 center error、IoU、coverage 等定量指标；需要先看预览图做视觉质控。
