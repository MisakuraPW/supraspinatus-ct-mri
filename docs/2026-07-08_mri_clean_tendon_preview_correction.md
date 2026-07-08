# 2026-07-08 MRI Clean Tendon Preview Correction

## Problem

The first MRI preview used a filled red overlay for `cor_ROI.nii.gz`. This was not a good way to inspect tendon appearance, because supraspinatus tendon on T2 fat-suppressed MRI is usually a low-signal dark band near the humeral head, and a filled overlay can hide the actual tissue signal.

The red overlay was not taken from the CT folder. It was loaded from each MR series directory:

```text
Data/label/<case>/MR/<series_uid>/cor_ROI.nii.gz
```

However, file location alone does not prove that `cor_ROI.nii.gz` is a precise MRI tendon segmentation. It may be a coarse ROI, a clinical marking, or a registered/converted label. Therefore it should be treated as a guide region unless annotation provenance is confirmed.

## Correction

Added a clean preview script:

- `scripts/render_mri_tendon_clean_previews.py`

It renders each key MRI slice as three panels:

1. raw MRI crop;
2. contrast-enhanced MRI crop;
3. MRI crop with only a thin ROI outline.

No filled red overlay is used.

## Command

```powershell
python scripts\render_mri_tendon_clean_previews.py --data-root Data\label --output-dir outputs\2026-07-08_mri_tendon_clean_previews
```

## Output

```text
outputs/2026-07-08_mri_tendon_clean_previews
```

Main contact sheet:

```text
outputs/2026-07-08_mri_tendon_clean_previews/mri_all_cases_clean_tendon_contact_sheet.png
```

Per-case files:

```text
outputs/2026-07-08_mri_tendon_clean_previews/cases/*_mri_clean_tendon_guide.png
```

## Visual Interpretation

- Left panel: raw MRI.
- Middle panel: contrast-enhanced MRI.
- Right panel: same MRI with green ROI outline and yellow ROI box.
- Green outline is only a guide from `cor_ROI.nii.gz`; it is not the tendon signal itself.

## Current Interpretation

The MRI tendon is not as visually obvious as CT bone. In these previews, the relevant structure should be searched as a dark, band-like or curved structure along the superior-lateral humeral head and below the acromion. The current ROI files can help point to the region, but should not be assumed to be a precise tendon segmentation until confirmed by the annotation source.
