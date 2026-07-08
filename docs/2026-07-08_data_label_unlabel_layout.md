# 2026-07-08 Data Layout Update

## Purpose

The project data layout was changed to make cloud upload and batch experiments cleaner:

```text
Data/
├── label/
│   └── <case>/
│       ├── CT/
│       └── MR/
└── unlabel/
    └── <case>/
        └── PA*/ST*/SE*/IM*
```

## Code Changes

Default data roots were updated:

- labeled CT locator: `Data/label`
- unlabeled DICOM inference: `Data/unlabel`
- TotalSegmentator shoulder-bone preprocessing: `Data/label`
- current-method preview rendering: `Data/label` and `Data/unlabel`
- bone-edge preview rendering: `Data/label` and `Data/unlabel`
- MRI viewer/static preview scripts: `Data/label`
- grid search: `Data/label`

The MRI series discovery logic was also updated so that passing either `Data` or `Data/label` can find nested `Data/label/<case>/MR` series. When scanning from `Data`, case names are inferred as `<case>` instead of `label`.

## Validation

Validated with:

```powershell
python -m py_compile scripts\run_multibone_locator.py scripts\run_multibone_dicom_inference.py scripts\run_totalseg_shoulder_bones.py scripts\render_current_method_previews.py scripts\render_bone_edge_tendon_previews.py scripts\run_mri_viewer.py scripts\render_mri_tendon_previews.py scripts\render_mri_tendon_clean_previews.py src\supraspinatus_locator\localization\multi_bone_traditional.py src\supraspinatus_locator\preprocessing\totalseg_bones.py
python scripts\run_totalseg_shoulder_bones.py --data-dir Data\label --output-dir outputs\_tmp_totalseg_dryrun --cases LHY --dry-run --fast --quiet
python scripts\run_mri_viewer.py --data-root Data\label --list --list-csv outputs\_tmp_mri_series_index.csv
```

Unlabeled case discovery found 10 cases under `Data/unlabel`.

## Cloud Notes

Upload data as one `Data.zip` with the same folder layout. Code and docs are committed to GitHub; data remains ignored by Git and should be uploaded separately to the cloud machine.
