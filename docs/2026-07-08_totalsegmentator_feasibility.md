# 2026-07-08 TotalSegmentator Feasibility For Shoulder Tendon Project

## Question

Assess whether `refercode/TotalSegmentator` can be applied to the current CT-MRI supraspinatus tendon localization/segmentation project.

## Local Source Checked

- `refercode/TotalSegmentator/README.md`
- `refercode/TotalSegmentator/AGENTS.md`
- `refercode/TotalSegmentator/totalsegmentator/registry.py`
- `refercode/TotalSegmentator/totalsegmentator/map_to_binary.py`

## Key Finding

TotalSegmentator is useful for this project, but mainly as a robust anatomy prior / bone segmentation provider, not as a direct tendon locator.

The open `total` CT task contains the shoulder-related bones:

- `humerus_left`
- `humerus_right`
- `scapula_left`
- `scapula_right`
- `clavicula_left`
- `clavicula_right`

The open `total_mr` MR task also contains the same shoulder-related bones.

The explicit shoulder muscle task exists and includes:

- `deltoid`
- `supraspinatus`
- `infraspinatus`
- `subscapularis`

However, `thigh_shoulder_muscles` and `thigh_shoulder_muscles_mr` require a TotalSegmentator license. Academic/non-commercial licenses are described by the upstream README as available separately.

## Recommended Use In This Project

Use TotalSegmentator first for CT bone segmentation:

```powershell
TotalSegmentator -i input_ct.nii.gz -o output_seg `
  -ta total `
  --roi_subset humerus_left humerus_right scapula_left scapula_right clavicula_left clavicula_right `
  --report output_seg\run_report.json `
  --statistics --statistics_extra
```

Then use these masks to replace or validate our current threshold-based bone segmentation:

- humeral head fitting from `humerus_left/right`;
- scapula/glenoid/acromion anchor from `scapula_left/right`;
- clavicle/acromioclavicular spatial context from `clavicula_left/right`;
- bone distance maps and surface arcs from clean model masks;
- sanity checks for wrong bone component selection.

## Expected Benefit

TotalSegmentator can directly address one weakness of the current morphology pipeline: our bone masks come from simple CT thresholding and connected components, so wrong component selection or incomplete shoulder-bone separation can affect candidate generation.

Replacing that stage with anatomical labels should make candidate generation more stable and interpretable:

- no need to guess whether a component is humerus vs scapula;
- side-specific labels help avoid wrong-side/wrong-bone errors;
- cleaner humeral head mask should improve sphere/circle fitting;
- better scapula and clavicle context should improve the supraspinatus corridor definition.

## Limitations

TotalSegmentator will not directly solve tendon localization in CT unless using the licensed shoulder muscle task, and even that task segments muscle belly rather than necessarily the tendon insertion.

Important risks:

- the full `total` model was built for broad CT anatomy, not specifically thin shoulder tendons;
- shoulder-only CT volumes may be partially cropped, which can reduce robustness;
- `--roi_subset` is faster but upstream notes it may be less accurate for some small structures;
- model inference needs PyTorch, nnU-Net dependencies, and pretrained weights;
- first run may need network access to download weights;
- licensed tasks cannot be assumed available in a clean cloud environment.

## Best Integration Strategy

Do not replace the whole locator with TotalSegmentator. Add it as an optional preprocessing backend:

1. Convert/select CT series to NIfTI.
2. Run TotalSegmentator `total` with shoulder-bone `--roi_subset`.
3. Save masks under a separate output directory.
4. Build a new candidate generator/source using these masks.
5. Compare against current morphology masks on the same old 10 labeled cases and new 10 visual cases.
6. Fall back to current threshold masks when TotalSegmentator fails or is unavailable.

## Short Conclusion

Yes, it can be applied well, but the right role is "segmented shoulder-bone anatomical prior" rather than "automatic tendon answer." It is especially promising for improving the multi-bone anchor and bone-edge tendon-channel methods.
