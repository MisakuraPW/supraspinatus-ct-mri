# 2026-07-08 TotalSegmentator Bone Backend Integration

## Purpose

Add an optional TotalSegmentator-based shoulder bone segmentation backend for the existing traditional CT supraspinatus localization pipeline.

This does not replace the whole locator. It only replaces the bone mask source used by the multi-bone / bone-edge candidate generation logic.

## New Files

- `src/supraspinatus_locator/preprocessing/totalseg_bones.py`
- `scripts/run_totalseg_shoulder_bones.py`

## Modified Files

- `src/supraspinatus_locator/localization/multi_bone_traditional.py`
- `scripts/run_multibone_locator.py`

## What Changed

The original method used:

```python
bone_mask = image > bone_threshold_hu
```

The locator now accepts an optional external bone mask. If provided, the external mask is used for:

- anchor candidate generation;
- bone edge candidate generation;
- bone overlap;
- near-bone / margin-bone / bone-distance features.

The default behavior is unchanged if no external bone mask is provided.

## TotalSegmentator Bone Generation

The recommended TotalSegmentator task is the open CT `total` task with shoulder bone `roi_subset`:

```powershell
python scripts\run_totalseg_shoulder_bones.py `
  --data-dir Data\label `
  --output-dir outputs\2026-07_totalseg_shoulder_bones `
  --fast `
  --device gpu
```

For CPU/cloud without GPU:

```powershell
python scripts\run_totalseg_shoulder_bones.py `
  --data-dir Data\label `
  --output-dir outputs\2026-07_totalseg_shoulder_bones `
  --fast `
  --device cpu
```

The script requests these classes:

- `humerus_left`
- `humerus_right`
- `scapula_left`
- `scapula_right`
- `clavicula_left`
- `clavicula_right`

For each case it writes:

```text
outputs/2026-07_totalseg_shoulder_bones/<case>/segmentations/*.nii.gz
outputs/2026-07_totalseg_shoulder_bones/<case>/shoulder_bones_combined.nii.gz
outputs/2026-07_totalseg_shoulder_bones/<case>/run_report.json
```

## Run Locator With TotalSegmentator Bone Masks

After the masks are generated:

```powershell
python scripts\run_multibone_locator.py `
  --data-dir Data\label `
  --output-dir outputs\2026-07_totalseg_bone_backend_locator `
  --bone-mask-dir outputs\2026-07_totalseg_shoulder_bones `
  --surface-arc-enable `
  --bone-edge-enable `
  --selection-policy generalized `
  --export-candidates
```

If some cases are missing TotalSegmentator masks and threshold fallback is acceptable:

```powershell
python scripts\run_multibone_locator.py `
  --data-dir Data\label `
  --output-dir outputs\2026-07_totalseg_bone_backend_locator `
  --bone-mask-dir outputs\2026-07_totalseg_shoulder_bones `
  --allow-threshold-bone-fallback
```

## Local Run Status

Local inference was stopped by user decision. The current local machine has CPU-only PyTorch, so full TotalSegmentator inference would likely be slow. The intended usage is cloud/GPU execution.

The code path was statically checked with `py_compile`; no valid TotalSegmentator segmentation result has been produced locally in this round.

## Notes

TotalSegmentator itself is a deep learning model based on nnU-Net. It can run on CPU, but practical use for multiple CT volumes should preferably use a GPU environment.

This integration keeps the morphology method structure intact while allowing a stronger learned bone segmentation backend.
