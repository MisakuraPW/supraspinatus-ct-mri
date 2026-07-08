# 2026-07-08 MRI Tendon Viewer

## Purpose

This update adds a lightweight MRI viewer entry point so we can inspect what the supraspinatus tendon region looks like on the MR series before building CT-MRI registration or multimodal segmentation code.

The script reuses the existing PyQt volume viewer and adds MRI-specific loading:

- discover MR DICOM series under `Data/label/*/MR`;
- open one MR DICOM series by case name or series index;
- open NIfTI MRI volumes directly;
- optionally overlay a NIfTI ROI/mask/prediction;
- automatically choose an MRI display window from image percentiles.

## New Script

- `scripts/run_mri_viewer.py`

## List Available MR Series

```powershell
python scripts\run_mri_viewer.py --data-root Data\label --list --list-csv outputs\mri_series_index.csv
```

This writes a series index table to:

```text
outputs/mri_series_index.csv
```

Current scan found 11 MR DICOM series, including HMC, LHY, LWL, OSQ, SB, WQX, YPL, ZH, ZJ, and ZJY.

## Open A Case

Open the automatically preferred T2 fat-suppressed coronal-like series:

```powershell
python scripts\run_mri_viewer.py --data-root Data\label --case LHY --mask auto
```

For HMC, two MR series were found. Open the second filtered series:

```powershell
python scripts\run_mri_viewer.py --data-root Data\label --case HMC --series-index 1 --mask auto
```

Filter by sequence text:

```powershell
python scripts\run_mri_viewer.py --data-root Data\label --case LHY --contains t2 --mask auto
```

Open a specific DICOM series directory:

```powershell
python scripts\run_mri_viewer.py --image "Data\LHY\MR\series_folder" --mask auto
```

Open a NIfTI MRI volume and ROI directly:

```powershell
python scripts\run_mri_viewer.py --image "Data\ZJY\MR\fixed.nii.gz" --mask "Data\ZJY\MR\cor_ROI.nii.gz"
```

## Viewer Controls

- Use the slice slider or mouse wheel to move through slices.
- Use the axis selector to switch axial/coronal/sagittal views.
- Red overlay is the mask/ROI if `--mask` is provided.
- Cyan overlay is the prediction if `--pred` is provided.
- If `--mask auto` finds a candidate ROI but its shape does not match the MRI volume, the script skips it and prints the mismatch instead of crashing.

## Dependencies

If the viewer environment is not installed:

```powershell
pip install -r requirements\base.txt
pip install -r requirements\viewer.txt
```

## Validation

Validated with:

```powershell
python -m py_compile scripts\run_mri_viewer.py
python scripts\run_mri_viewer.py --data-root Data\label --list --list-csv outputs\mri_series_index.csv
```

The GUI was not launched during validation to avoid blocking the coding session.
