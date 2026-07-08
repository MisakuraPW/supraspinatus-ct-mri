# LHY 数据初步记录

当前观察到的样例结构：

- `LHY/LHY/CT/`：包含 CT DICOM 序列 `SE0` 到 `SE7`，以及多种 `.nii.gz` 体数据。
- `LHY/LHY/CT/ROI.nii.gz`：二值 ROI，适合用于第一阶段候选 ROI 覆盖率验证。
- `LHY/LHY/MR/.../`：包含 MR DICOM 和 `cor_ROI.nii.gz`。

初步建议：

- 先使用 `60kev.nii.gz` 或 `70kev.nii.gz` 跑规则定位 baseline。
- 用 `ROI.nii.gz` 验证 ROI recall 与 bbox IoU。
- 后续若要做 CT-MRI 弱监督，需要额外确认 CT/MR 是否同患者、是否有可靠空间配准关系。

