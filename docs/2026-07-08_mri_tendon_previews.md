# 2026-07-08 MRI Tendon Preview Generation

## Purpose

This update adds static preview generation for the MRI tendon annotations. The goal is to inspect what the supraspinatus tendon/ROI looks like on MRI without launching the interactive PyQt viewer.

## New Script

- `scripts/render_mri_tendon_previews.py`

## Command

```powershell
python scripts\render_mri_tendon_previews.py --data-root Data\label --output-dir outputs\2026-07-08_mri_tendon_previews
```

Optional single-case rendering:

```powershell
python scripts\render_mri_tendon_previews.py --data-root Data\label --case LHY --output-dir outputs\2026-07-08_mri_tendon_previews_LHY
```

## Output

Main output directory:

```text
outputs/2026-07-08_mri_tendon_previews
```

Generated files:

- `mri_all_cases_all_slices_contact_sheet.png`: all-slice contact sheet for all cases.
- `mri_all_cases_tendon_focus_contact_sheet.png`: ROI-focused contact sheet for all cases.
- `mri_tendon_preview_summary.csv`: per-case MRI series, ROI status, ROI voxel count, and preview paths.
- `cases/*_mri_all_slices.png`: one all-slice montage per case.
- `cases/*_mri_tendon_focus.png`: one cropped tendon/ROI-focused montage per case.

## Visual Legend

- Grayscale image: MRI slice.
- Red transparent overlay: `cor_ROI.nii.gz`.
- Yellow box: 2D bounding box of the ROI on that slice.
- Yellow text: case name, slice index, and ROI note.

## Notes

- The script discovers MR DICOM series under `Data/label/*/MR`.
- For each case, it chooses the preferred coronal T2 fat-suppressed MR series through the same discovery logic used by `scripts/run_mri_viewer.py`.
- If an ROI shape matches the image shape, it is overlaid directly.
- If the ROI first two dimensions are transposed relative to the DICOM series, the script automatically transposes x/y. This occurred for OSQ.
- If an ROI is missing or shape-incompatible, the image preview is still generated and the status is recorded in the summary CSV.

## Validation

Validated with:

```powershell
python -m py_compile scripts\render_mri_tendon_previews.py
python scripts\render_mri_tendon_previews.py --data-root Data\label --output-dir outputs\2026-07-08_mri_tendon_previews
```

The run generated previews for 10 MRI cases.
