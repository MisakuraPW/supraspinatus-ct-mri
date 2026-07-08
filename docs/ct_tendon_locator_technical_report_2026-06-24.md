# CT 冈上肌腱采样 ROI 定位技术报告

日期：2026-06-24  
数据：10 例肩关节 CT/MRI 数据  
任务：在 CT 中自动定位冈上肌腱相关采样 ROI，而不是完整肌腱分割

## 1. 任务背景

冈上肌腱在常规 CT 中软组织边界不清晰，直接做完整肌腱分割风险较高。因此当前阶段采用更现实的任务定义：在 60keV CT 中自动生成一个位于冈上肌腱走行区域附近、避开骨皮质的采样 ROI，用于后续影像特征提取、MRI 分割对照和胶原含量/分级预测。

当前 CT 医生 ROI 应理解为“采样 ROI”或“参考采样区域”，不是完整冈上肌腱真值。因此评价时不能只按 Dice 或完全重合判断，而应同时关注：

- ROI 是否落在合理解剖软组织区。
- ROI 是否避免压入骨皮质。
- ROI 与医生采样 ROI 的空间关系是否更接近。
- 方法是否稳定适用于多病例。

## 2. 数据与输入

本轮实验使用 `Data/` 下 10 例数据：

| 病例 |
|---|
| HMC |
| LHY |
| LWL |
| OSQ |
| SB |
| WQX |
| YPL |
| ZH |
| ZJ |
| ZJY |

每例主要使用：

- CT：`60kev.nii.gz` 或 `60KEV.nii.gz`
- 医生 CT 采样 ROI：`roi.nii.gz`、`ROI.nii.gz`、`roi.nii` 或 `ROI.nii`

为兼容老师代码的 `case/CT/60keV + roi` 输入结构，已将 10 例轻量归一化到：

```text
outputs/normalized_10cases/
```

该目录只用于运行实验，不改变原始数据。

## 3. 对比方法

### 3.1 老师方法：单上外侧肱骨锚点

老师方法是 CT-only 传统图像处理方法，核心流程如下：

1. 读取 60keV CT。
2. 使用 `60keV > 300 HU` 提取骨皮质。
3. 在候选层上做 2D 骨连通域分析。
4. 根据面积、宽高、位置、侧别等规则筛选近端肱骨连通域。
5. 从近端肱骨连通域中裁出上外侧局部锚点。
6. 在锚点上方/外侧生成候选采样 ROI。
7. 对候选 ROI 执行硬过滤：
   - 骨重叠不能过高。
   - ROI 不能落到体外。
   - 平均 CT 值要符合软组织范围。
8. 对有效候选打分并选择最高分 ROI。

该方法的优点是实现精细、可解释、骨重叠控制强。局限是主要依赖止点附近的上外侧肱骨锚点，能够知道“止点附近在哪里”，但没有显式建模冈上肌腱从肩胛冈上窝到大结节的走行方向。

### 3.2 本次方法：多骨锚点传统定位

本次实现的方法保留老师的评价标准和主要约束，但加入多骨锚点思想。核心变化是：

- 保留近端肱骨上外侧锚点，作为止点/大结节附近参考。
- 新增肩峰/肩胛顶板骨组件，作为肩峰下通道上界参考。
- 将候选 ROI 放在“肱骨上外侧锚点”和“肩峰/肩胛顶板”共同限定的软组织通道中。
- 继续使用老师同类硬过滤：骨重叠、体内比例、软组织 CT 值。

因此本方法不再只回答“止点附近在哪里”，而是尝试进一步回答：

```text
肌腱大致从哪里来、经过哪个肩峰下通道、往哪里止点走。
```

实现文件：

```text
src/supraspinatus_locator/localization/multi_bone_traditional.py
scripts/run_multibone_locator.py
scripts/compare_teacher_multibone.py
```

运行命令：

```powershell
python teacher\code\code\ct_tendon_locator_package\run_pipeline.py --data-dir outputs\normalized_10cases --output-dir outputs\teacher_10cases
python scripts\run_multibone_locator.py --data-dir outputs\normalized_10cases --output-dir outputs\multibone_10cases
python scripts\compare_teacher_multibone.py --teacher-csv outputs\teacher_10cases\evaluation\ct_tendon_locator_results.csv --multibone-csv outputs\multibone_10cases\multibone_locator_results.csv --out outputs\multibone_10cases\teacher_vs_multibone.csv
```

## 4. 评价指标

本轮使用与老师方法兼容的指标：

| 指标 | 含义 | 越大/越小越好 |
|---|---|---|
| `center_error_mm` | 算法 ROI 中心到医生 ROI 中心的物理距离 | 越小越好 |
| `pred_box_doctor_bbox_iou` | 算法 ROI bbox 与医生 ROI bbox 的 3D IoU | 越大越好 |
| `doctor_roi_coverage` | 医生 ROI 体素被算法 ROI 覆盖的比例 | 越大越好 |
| `pred_bone_overlap` | 算法 ROI 中 `>300 HU` 骨皮质体素占比 | 越小越好 |

注意：医生 ROI 不是完整肌腱真值，所以 `doctor_roi_coverage` 和 bbox IoU 只能作为“参考采样区域接近程度”，不能作为唯一目标。`pred_bone_overlap` 对 CT 采样 ROI 特别重要，因为 ROI 压入骨头会污染后续特征。

## 5. 总体结果

| 指标 | 老师单锚点方法 | 多骨锚点方法 | 变化 |
|---|---:|---:|---:|
| 平均中心误差 mm | 7.771 | 6.311 | -1.460 |
| 中心误差中位数 mm | 7.295 | 5.225 | -2.070 |
| 平均 bbox IoU | 0.0578 | 0.1089 | +0.0511 |
| bbox IoU 中位数 | 0.0450 | 0.1125 | +0.0675 |
| 平均医生 ROI coverage | 0.0851 | 0.1855 | +0.1004 |
| 医生 ROI coverage 中位数 | 0.0475 | 0.1886 | +0.1411 |
| 平均骨重叠 | 0.0006 | 0.0029 | +0.0023 |

逐指标胜负：

| 指标 | 多骨方法改善病例数 |
|---|---:|
| 中心误差 | 7/10 |
| bbox IoU | 6/10 |
| 医生 ROI coverage | 7/10 |
| 骨重叠不高于老师方法 | 3/10 |

整体看，多骨锚点方法在空间接近性指标上明显优于老师单锚点方法，但骨重叠略有增加。不过多骨方法平均骨重叠 `0.0029`，仍低于老师代码中的硬约束阈值 `0.012`。

## 6. 逐例结果

| 病例 | 老师中心误差 | 多骨中心误差 | 老师 IoU | 多骨 IoU | 老师 coverage | 多骨 coverage | 老师骨重叠 | 多骨骨重叠 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HMC | 5.95 | 5.81 | 0.0636 | 0.0636 | 0.0540 | 0.0601 | 0.0000 | 0.0000 |
| LHY | 5.59 | 3.72 | 0.0908 | 0.1127 | 0.1432 | 0.1844 | 0.0000 | 0.0016 |
| LWL | 6.53 | 5.50 | 0.1264 | 0.1644 | 0.0411 | 0.1929 | 0.0000 | 0.0000 |
| OSQ | 8.06 | 10.95 | 0.0368 | 0.0000 | 0.2114 | 0.0000 | 0.0018 | 0.0037 |
| SB | 8.02 | 10.63 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0024 | 0.0031 |
| WQX | 11.63 | 9.96 | 0.0000 | 0.0593 | 0.0000 | 0.0263 | 0.0000 | 0.0005 |
| YPL | 6.78 | 4.68 | 0.0532 | 0.1123 | 0.1448 | 0.2801 | 0.0000 | 0.0102 |
| ZH | 13.07 | 2.74 | 0.0000 | 0.2993 | 0.0000 | 0.5740 | 0.0005 | 0.0000 |
| ZJ | 4.27 | 4.95 | 0.2071 | 0.1550 | 0.2566 | 0.2473 | 0.0016 | 0.0086 |
| ZJY | 7.81 | 4.17 | 0.0000 | 0.1227 | 0.0000 | 0.2896 | 0.0000 | 0.0010 |

## 7. 结果分析

### 7.1 明显提升病例

多骨方法在 ZH、ZJY、YPL、LWL、LHY 上提升明显。

| 病例 | 主要提升 |
|---|---|
| ZH | coverage 从 0.0000 提升到 0.5740，中心误差从 13.07 mm 降到 2.74 mm |
| ZJY | coverage 从 0.0000 提升到 0.2896，中心误差从 7.81 mm 降到 4.17 mm |
| YPL | coverage 从 0.1448 提升到 0.2801，中心误差从 6.78 mm 降到 4.68 mm |
| LWL | coverage 从 0.0411 提升到 0.1929，bbox IoU 从 0.1264 提升到 0.1644 |
| LHY | coverage 从 0.1432 提升到 0.1844，中心误差从 5.59 mm 降到 3.72 mm |

这些病例说明，多骨锚点能够修正单肱骨锚点对肌腱走行方向理解不足的问题。当肩峰/肩胛顶板组件可被稳定识别时，算法 ROI 更容易落入肩峰下软组织通道。

### 7.2 退步或失败病例

| 病例 | 问题 |
|---|---|
| OSQ | 多骨方法 coverage 从 0.2114 降到 0.0000，是当前最明显失败例 |
| SB | 两种方法 coverage 都为 0，多骨中心误差更大 |
| ZJ | 多骨略退步，coverage 从 0.2566 降到 0.2473，骨重叠升高 |

OSQ 说明当前多骨规则仍可能选错肩峰/肩胛顶板组件，或者在顶板和肱骨之间生成的 corridor 方向不符合该病例实际标注位置。SB 说明单纯骨性锚点仍不足以覆盖所有病例，可能需要更强的切片连续性、左右方向归一化或医生标注层面复核。

### 7.3 骨重叠风险

多骨方法平均骨重叠从 `0.0006` 升高到 `0.0029`，说明它为了更贴近医生 ROI，有时会更靠近骨边缘。虽然仍低于硬阈值 `0.012`，但后续优化时应把骨重叠作为强约束继续压低，尤其关注 YPL 和 ZJ：

| 病例 | 多骨骨重叠 |
|---|---:|
| YPL | 0.0102 |
| ZJ | 0.0086 |
| OSQ | 0.0037 |
| SB | 0.0031 |

## 8. 当前结论

本轮实验支持“多骨锚点传统图像处理方法优于单上外侧肱骨锚点”的初步判断。

主要证据：

- 平均中心误差降低约 `1.46 mm`。
- 平均 bbox IoU 约提升 `88%`，从 `0.0578` 到 `0.1089`。
- 平均医生 ROI coverage 约提升 `118%`，从 `0.0851` 到 `0.1855`。
- 10 例中 7 例中心误差改善，7 例 coverage 改善。

但该方法还没有达到稳定可交付状态。当前主要问题是：

- OSQ 明显失败。
- SB 两种方法都未覆盖医生 ROI。
- 多骨方法骨重叠略升高。
- 当前规则仍依赖 2D 连通域、固定阈值和经验参数，对体位、FOV、骨连接形态敏感。

## 9. 下一步建议

优先级从高到低：

1. 复盘 OSQ 失败例，检查多骨方法选中的顶板组件是否错误。
2. 对 SB 做单例分析，判断是算法问题还是医生采样 ROI 与骨性先验关系不一致。
3. 增加切片连续性约束，避免单层组件偶然误选。
4. 将多骨候选与老师单锚点候选做融合：当多骨评分低或骨重叠偏高时，回退到老师候选。
5. 增加候选集输出，而不是只输出 top-1 ROI，让医生审阅 top-3/top-5。
6. 后续再考虑把肩峰、肱骨头、大结节、关节盂等结构升级为显式 3D 锚点。

## 10. 输出文件索引

```text
outputs/teacher_10cases/evaluation/ct_tendon_locator_results.csv
outputs/teacher_10cases/evaluation/ct_tendon_locator_summary.txt
outputs/teacher_10cases/evaluation/ct_tendon_locator_contact_sheet.png
outputs/teacher_10cases/evaluation/ct_tendon_locator_3d_bbox_contact_sheet.png

outputs/multibone_10cases/multibone_locator_results.csv
outputs/multibone_10cases/multibone_locator_summary.txt
outputs/multibone_10cases/teacher_vs_multibone.csv
outputs/multibone_10cases/*_multibone_preview.png
outputs/multibone_10cases/*_multibone_roi.nii.gz
```

