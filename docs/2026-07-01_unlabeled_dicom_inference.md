# 2026-07-01 无标注 DICOM 10 例推理记录

## 改动内容

- 新增 `scripts/run_multibone_dicom_inference.py`。
- 目的：在没有医生冈上肌腱 ROI 标注的新增 CT DICOM 病例上，直接运行当前最佳的多骨 + 肱骨头弧形候选定位方法。
- 输入结构：每个病例目录下为 `PA*/ST*/SE*` DICOM 序列；脚本会自动选择 `SeriesDescription` 中匹配 `Monochromatic 60 Kev` 的序列。
- 输出内容：每例预测 ROI NIfTI、每例预览图、top-k 候选 CSV、最终候选 CSV、无标注内部自检摘要。

## 本次运行

数据目录：`Data/260626-ct-10例`

输出目录：`outputs/2026-07-01_unlabeled_10cases_inference`

运行命令：

```powershell
python scripts\run_multibone_dicom_inference.py --data-dir Data\260626-ct-10例 --output-dir outputs\2026-07-01_unlabeled_10cases_inference --current-anchor-count 160 --bone-margin-voxels 2 --continuity-window 2 --continuity-xy-tolerance 14 --low-z-enable --low-z-range-mm 12 --low-z-step-mm 2 --low-z-weight 0.85 --branch-anchor-count 6 --surface-arc-enable --surface-arc-select-enable --surface-arc-anchor-count 8 --surface-arc-weight 0.92 --surface-arc-angle-min-deg 25 --surface-arc-angle-max-deg 82 --surface-arc-angle-step-deg 14 --surface-arc-offset-min-voxels 4 --surface-arc-offset-max-voxels 18 --surface-arc-offset-step-voxels 4 --surface-arc-z-window 1 --surface-arc-max-bone-fraction 0.12 --surface-arc-target-bone-fraction 0.035 --surface-arc-bone-sigma 0.045 --surface-arc-target-offset-voxels 10 --surface-arc-offset-sigma-voxels 9 --surface-arc-sphere-blend 0 --topk 5
```

## 结果摘要

无医生标注，因此本次不能计算 center error、IoU、doctor ROI coverage。这里只能报告内部自检指标。

| 指标 | 数值 |
|---|---:|
| 成功病例数 | 10 |
| 失败病例数 | 0 |
| 平均骨重叠 | 0.0110 |
| 最大骨重叠 | 0.0562 |
| 平均近骨比例 | 0.0196 |
| 平均边界贴骨比例 | 0.0308 |
| 最终候选来源 | current_multibone: 8; surface_arc: 2 |

## 逐例结果

| 病例 | 最终方法 | 决策原因 | 分数 | 骨重叠 | 近骨比例 | 边界贴骨比例 | 中心 x | 中心 y | 中心 z |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 刘春雷 | current_multibone | choose_multibone_high_confidence | 4.5758 | 0.0000 | 0.0000 | 0.0075 | 192.00 | 173.76 | 30.00 |
| 周洁 | surface_arc | choose_surface_arc_humeral_head_candidate | 3.9304 | 0.0562 | 0.0602 | 0.0583 | 368.61 | 209.47 | 32.00 |
| 唐解成 | surface_arc | choose_surface_arc_humeral_head_candidate | 5.2825 | 0.0481 | 0.0449 | 0.0447 | 135.17 | 212.94 | 27.00 |
| 张秋燕 | current_multibone | choose_multibone_high_confidence | 4.6344 | 0.0000 | 0.0016 | 0.0109 | 178.50 | 182.86 | 26.00 |
| 杨国玲 | current_multibone | choose_multibone_high_confidence | 4.3569 | 0.0000 | 0.0000 | 0.0000 | 335.00 | 205.58 | 33.00 |
| 杨青玉 | current_multibone | choose_multibone_high_confidence | 5.1884 | 0.0000 | 0.0062 | 0.0195 | 161.50 | 177.90 | 25.00 |
| 王小花 | current_multibone | choose_multibone_high_confidence | 4.3055 | 0.0000 | 0.0099 | 0.0270 | 348.00 | 202.88 | 22.00 |
| 王钦正 | current_multibone | choose_multibone_high_confidence | 3.9784 | 0.0058 | 0.0359 | 0.0482 | 353.00 | 193.24 | 30.00 |
| 胡思田 | current_multibone | choose_multibone_high_confidence | 3.4992 | 0.0000 | 0.0177 | 0.0548 | 186.00 | 183.84 | 37.00 |
| 郝慧 | current_multibone | choose_multibone_high_confidence | 4.8272 | 0.0000 | 0.0198 | 0.0370 | 310.00 | 182.20 | 31.00 |

## 注意事项

- 本次新增病例没有医生标注，所以不能判断真正定位误差；预览图是主要人工质检依据。
- 当前脚本在无 teacher CSV 时不会触发依赖老师 z 中心的 `teacher_z_refine/contact_z` 分支；主要使用 `current_multibone` 和 `surface_arc`。
- 如果后续医生补标 ROI，可以直接用有标注评估入口重算 center error、IoU、ROI coverage，并与本次无标注推理结果对照。
