# 2026-07-06 Current Method Preview Generation

## 目的

为当前稳健化方法生成旧 10 例和新 10 例的预览图，便于快速目测定位效果。

## 当前方法定义

本轮预览使用上一轮候选池排序结果：

- 旧 10 例：`outputs/2026-07_candidate_ranker_experiment/results/unified_learned_labeled_selections.csv`
- 新 10 例：`outputs/2026-07_candidate_ranker_experiment_with_visual_feedback/results/unlabeled_unified_selections.csv`

旧 10 例预览中：

- 青色框：当前方法预测 ROI
- 红色区域：医生 ROI 标注

新 10 例预览中：

- 青色框：当前方法预测 ROI
- 因无医生标注，不显示红色区域

## 新增脚本

新增：

`scripts/render_current_method_previews.py`

用途：

- 读取当前排序结果 CSV
- 读取旧 10 例 NIfTI CT 和医生 ROI
- 读取新 10 例 DICOM CT
- 每例输出中心层及前后 2 层的三切片预览
- 同时输出旧 10 例和新 10 例总览拼图

## 输出位置

输出目录：

`outputs/2026-07-06_current_method_previews`

关键文件：

- `outputs/2026-07-06_current_method_previews/labeled_current_method_contact_sheet.png`
- `outputs/2026-07-06_current_method_previews/unlabeled_current_method_contact_sheet.png`
- `outputs/2026-07-06_current_method_previews/labeled/*_current_method_preview.png`
- `outputs/2026-07-06_current_method_previews/unlabeled/*_current_method_preview.png`

## 运行命令

```powershell
python scripts\render_current_method_previews.py --output-dir outputs\2026-07-06_current_method_previews
```

本次已生成：

- 旧 10 例单例预览：10 张
- 新 10 例单例预览：10 张
- 旧 10 例总览拼图：1 张
- 新 10 例总览拼图：1 张
