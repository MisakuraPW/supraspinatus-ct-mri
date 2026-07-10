# TotalSeg 骨分割非训练优化代码记录

本次新增的目标不是训练骨分割模型，而是把现有 TotalSegmentator 输出做成更可靠的“肩部骨骼先验”，为后续传统定位策略提供更稳定的骨 mask。

## 新增脚本

### `scripts/fuse_totalseg_hu_bone_masks.py`

功能：

- 读取一个或多个 TotalSeg 输出目录。
- 把 fast/full-res 等多次 TotalSeg 的肩部骨 mask 合并成 `TotalSeg prior`。
- 用 `HU > 300` 提取 CT 皮质骨候选。
- 生成 `TotalSeg-guided HU`：只在 TotalSeg 先验邻域内保留 HU 骨体素。
- 对融合 mask 做连通域清理和质量评分。
- 输出融合 mask、预览图、CSV 总表和报告。

关键输出：

- `<output>/<case>/shoulder_bones_totalseg_prior_union.nii.gz`
- `<output>/<case>/shoulder_bones_hu_threshold.nii.gz`
- `<output>/<case>/shoulder_bones_totalseg_guided_hu_raw.nii.gz`
- `<output>/<case>/shoulder_bones_fused_hu.nii.gz`
- `<output>/totalseg_hu_fusion_summary.csv`
- `<output>/reports/totalseg_hu_fusion_report.md`
- `<output>/previews/*_fusion_preview.png`

如果某例 TotalSeg 明显失败，`shoulder_bones_fused_hu.nii.gz` 会输出为空 mask。后续定位脚本配合 `--allow-threshold-bone-fallback` 可以自动回退到纯 HU，不会强行相信错误分割。

### `scripts/run_totalseg_bone_segmentation_suite.py`

功能：

- 云端一键运行 fast TotalSeg、full-res TotalSeg。
- 自动把两套输出送入融合脚本。
- 打印下一步定位 sweep 的推荐命令。

默认情况下，如果不指定 `--run-fast/--run-fullres/--skip-totalseg`，会同时跑 fast 和 full-res。

## 云端推荐运行方式

先保证 TotalSegmentator 和 PyTorch 环境已经可用。

```bash
python scripts/run_totalseg_bone_segmentation_suite.py \
  --data-dir Data/label \
  --output-root outputs/2026-07_totalseg_bone_segmentation_suite \
  --device gpu \
  --require-gpu \
  --skip-existing
```

如果只想用 CPU，去掉 `--require-gpu` 并改成：

```bash
python scripts/run_totalseg_bone_segmentation_suite.py \
  --data-dir Data/label \
  --output-root outputs/2026-07_totalseg_bone_segmentation_suite_cpu \
  --device cpu \
  --skip-existing
```

如果已经分别跑过 fast/full-res，只做融合：

```bash
python scripts/run_totalseg_bone_segmentation_suite.py \
  --data-dir Data/label \
  --output-root outputs/2026-07_totalseg_bone_segmentation_suite \
  --skip-totalseg \
  --fast-dir outputs/2026-07_totalseg_shoulder_bones \
  --fullres-dir outputs/2026-07_totalseg_shoulder_bones_fullres
```

## 无标注 DICOM 数据准备

如果要对 `Data/unlabel` 里的 DICOM 病例跑同一套流程，先把每个患者自动选出的 CT 序列转成项目兼容的 NIfTI 文件夹：

```bash
python scripts/prepare_unlabeled_ct_nifti.py \
  --data-dir Data/unlabel \
  --output-dir outputs/2026-07_unlabel_ct_nifti
```

之后把 `--data-dir` 指向这个输出目录即可：

```bash
python scripts/run_totalseg_bone_segmentation_suite.py \
  --data-dir outputs/2026-07_unlabel_ct_nifti \
  --output-root outputs/2026-07_unlabel_totalseg_bone_segmentation_suite \
  --device gpu \
  --require-gpu \
  --skip-existing
```

注意：这个转换脚本只是为了让云端分割流程稳定吃到 NIfTI 输入；它会自动挑选每个病例中最像 CT 的 DICOM 序列，并生成预览图供肉眼确认。若预览显示选错序列，应先修正输入序列选择，再判断 TotalSeg 效果。

脚本会记录 `raw_spacing / spacing / z_spacing_source`。如果 DICOM 位置字段推出来的 z spacing 明显离谱，会自动回退到 `SliceThickness`，避免 TotalSeg 因体素尺度错误而失败。

## 后续定位推荐

融合完成后，用融合后的 HU 骨 mask 跑定位策略 sweep：

```bash
python scripts/run_totalseg_locator_strategy_sweep.py \
  --data-dir Data/label \
  --bone-mask-dir outputs/2026-07_totalseg_bone_segmentation_suite/totalseg_hu_fused \
  --bone-mask-filename shoulder_bones_fused_hu.nii.gz \
  --output-root outputs/2026-07_totalseg_fused_locator_sweep \
  --quick
```

## 设计理解

TotalSegmentator 的输出是语义骨结构，不一定等价于我们定位算法需要的“骨皮质边缘”。所以这里不直接拿 TotalSeg 替换骨阈值，而是让 TotalSeg 决定“肩部骨骼大致在哪里”，再用 HU 阈值决定“真正的高密度骨边缘在哪里”。这样可以减少胸廓、床板、噪声等非目标高密度结构干扰，同时保留 CT 皮质骨边缘信息。
