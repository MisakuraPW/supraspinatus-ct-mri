# 冈上肌腱 CT 自动定位与多模态分割研究原型

本项目用于探索肩关节 CT/MRI 多模态影像中的冈上肌腱候选区域自动定位，并为后续分割、配准迁移和胶原含量预测保留可扩展代码框架。当前第一步聚焦 **CT 中自动定位冈上肌腱候选 ROI**，不是精细分割。

## 文件架构

```text
.
├── README.md
├── .gitignore
├── requirements/
│   ├── base.txt
│   ├── viewer.txt
│   └── train.txt
├── configs/
│   ├── viewer.yaml
│   ├── locator_rule_based.yaml
│   └── train_roi_net.yaml
├── src/
│   └── supraspinatus_locator/
│       ├── data/              # DICOM/NIfTI 读取、数据扫描
│       ├── viewer/            # 体数据查看器与叠加显示
│       ├── preprocessing/     # CT 窗宽窗位、骨 mask、重采样
│       ├── localization/      # 规则定位、解剖先验、ROI 几何
│       ├── models/            # 深度学习路线骨架
│       ├── training/          # 训练/推理入口
│       ├── evaluation/        # ROI 覆盖率、IoU、缩小倍率等指标
│       └── utils/
├── scripts/
│   ├── scan_lhy_dataset.py
│   ├── run_viewer.py
│   ├── run_rule_based_locator.py
│   ├── view_localization.py
│   └── export_case_summary.py
├── docs/                      # 本地研究记录，默认被 git 忽略
├── outputs/                   # 实验输出，默认被 git 忽略
└── tests/
```

## 当前数据理解

当前推荐数据结构为 `Data/label` 与 `Data/unlabel`：

- CT：DICOM 序列 `SE0` 到 `SE7`，每个序列约 75 张；另有 `60kev.nii.gz`、`70kev.nii.gz`、`S1(Water).nii.gz`、`Muscle(Fat).nii.gz`、`eff.nii.gz` 和 `ROI.nii.gz`。
- MR：冠状位 T2 FS DICOM 序列与 `cor_ROI.nii.gz`。
- 阶段一优先使用 CT 体数据和 `ROI.nii.gz` 做候选 ROI 定位与覆盖率验证。

## 快速开始

建议先建虚拟环境，再安装基础依赖：

```bash
pip install -r requirements/base.txt
pip install -r requirements/viewer.txt
```

扫描 LHY 数据：

```bash
python scripts/scan_lhy_dataset.py --root Data/label/LHY --out outputs/summaries/lhy_summary.json
```

打开基础查看器：

```bash
python scripts/run_viewer.py --image Data/label/LHY/CT/60kev.nii.gz --mask Data/label/LHY/CT/ROI.nii.gz
```

运行规则定位 baseline：

```bash
python scripts/run_rule_based_locator.py --image Data/label/LHY/CT/60kev.nii.gz --target-mask Data/label/LHY/CT/ROI.nii.gz --out-dir outputs/localization/lhy_60kev
```

查看定位结果：

```bash
python scripts/view_localization.py --image Data/label/LHY/CT/60kev.nii.gz --mask Data/label/LHY/CT/ROI.nii.gz --pred outputs/localization/lhy_60kev/roi_mask.nii.gz
```

生成路线 C 的骨 mask、距离场和坐标先验通道：

```bash
python scripts/prepare_prior_channels.py --image Data/label/LHY/CT/60kev.nii.gz --out-dir outputs/priors/lhy_60kev
```

生成路线 E 的中心线热图：

```bash
python scripts/prepare_centerline_heatmap.py --image Data/label/LHY/CT/60kev.nii.gz --roi-json outputs/localization/lhy_60kev/roi.json --out outputs/localization/lhy_60kev/centerline_heatmap.nii.gz
```

运行传统多骨锚点定位器，并与老师单锚点方法的 CSV 对比：

```bash
python scripts/run_multibone_locator.py --data-dir Data/label --output-dir outputs/multibone_labeled
python scripts/compare_teacher_multibone.py --teacher-csv outputs/teacher_lhy/evaluation/ct_tendon_locator_results.csv --multibone-csv outputs/multibone_lhy/multibone_locator_results.csv --out outputs/multibone_lhy/teacher_vs_multibone.csv
```

## 方法路线

当前实现按调研结论分层推进：

1. 规则法 baseline：用 CT 骨阈值、连通域和物理尺寸先验生成冈上肌腱候选 ROI。
2. 传统多骨锚点路线：在老师上外侧肱骨锚点基础上，引入肩峰/肩胛顶板组件，约束肩峰下通道。
3. 骨骼先验路线：后续可替换为肩胛骨/肱骨分割、关键点提取和局部坐标系。
3. 深度学习路线：提供 ROI U-Net、距离场先验网络、中心线热图网络骨架，用于云端训练。
4. CT-MRI 弱监督路线：提供 SimpleITK 刚性配准脚本骨架，用于把 MRI ROI 粗迁移到 CT。
5. 评估逻辑：第一阶段主指标是 ROI recall、coverage、bbox IoU 和搜索空间缩小倍率，而不是 tendon Dice。
