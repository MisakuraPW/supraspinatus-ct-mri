# GitHub 与云端运行指导

## 目标

把本项目整理成适合上传 GitHub 的代码仓库，然后在云端 GPU 环境中上传数据、运行 TotalSegmentator 肩部骨分割，并用它替换当前传统方法中的 CT 阈值骨分割。

## 一、本地 Git 状态

本项目已经是一个 Git 仓库。

本轮已经调整 `.gitignore`，默认不上传：

- `Data/`
- `LHY/`
- `outputs/`
- `runs/`
- `checkpoints/`
- `wandb/`
- `refercode/`
- 模型权重：`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`
- 本地 Office/PDF 资料：`*.docx`, `*.pdf`

会上传：

- `src/`
- `scripts/`
- `configs/`
- `requirements/`
- `tests/`
- `docs/`
- `README.md`
- `.gitignore`

也就是说：GitHub 仓库主要放代码和说明文档，不放患者数据、实验输出、模型权重和外部参考项目。

## 二、创建 GitHub 仓库并关联

你在 GitHub 网页上新建一个空仓库，例如：

```text
supraspinatus-ct-mri
```

不要勾选自动生成 README、`.gitignore` 或 license，因为本地已经有这些文件。

然后在本地项目根目录运行：

```powershell
git remote add origin https://github.com/<你的用户名>/supraspinatus-ct-mri.git
git branch -M main
```

检查 remote：

```powershell
git remote -v
```

## 三、本地首次提交并推送

先检查将要提交的内容：

```powershell
git status --short
```

如果确认只包含代码和文档，不包含 `Data/`、`outputs/`、`refercode/` 等目录，则执行：

```powershell
git add .gitignore README.md configs requirements scripts src tests docs
git commit -m "Initialize supraspinatus CT-MRI localization project"
git push -u origin main
```

如果 Git 提示没有配置用户名和邮箱，执行：

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

如果只想对当前仓库配置，不影响其他项目：

```powershell
git config user.name "你的名字"
git config user.email "你的邮箱"
```

## 四、云端拉取代码

在云服务器或云算力平台中新建环境后：

```bash
git clone https://github.com/<你的用户名>/supraspinatus-ct-mri.git
cd supraspinatus-ct-mri
```

建议使用 Python 3.10 或 3.11。

创建环境示例：

```bash
conda create -n supraspinatus python=3.10 -y
conda activate supraspinatus
```

安装基础依赖：

```bash
pip install -r requirements/base.txt
pip install -r requirements/viewer.txt
pip install -r requirements/train.txt
```

安装 TotalSegmentator：

```bash
pip install TotalSegmentator
```

如果云端有 NVIDIA GPU，请安装与你的 CUDA 版本匹配的 PyTorch。以 CUDA 12.1 为例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 五、上传数据到云端

保持数据结构与本地一致。推荐云端放置为：

```text
Data/
├── label/
│   └── <case>/CT/60kev.nii.gz
│   └── <case>/CT/roi.nii.gz
│   └── <case>/MR/...
└── unlabel/
    └── <case>/PA*/ST*/SE*/IM*
```

旧 10 例标注数据建议放在：

```text
Data/label/<case>/CT/60kev.nii.gz
Data/label/<case>/CT/roi.nii.gz
```

如果云平台支持网页上传，直接上传压缩包后解压。命令行示例：

```bash
unzip Data.zip -d .
```

如果用 `scp`：

```bash
scp Data.zip <user>@<server>:/path/to/supraspinatus-ct-mri/
```

## 六、运行 TotalSegmentator 肩部骨分割

TotalSegmentator 是深度学习模型，适合云端 GPU 跑。

运行旧 10 例：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --fast \
  --device gpu
```

如果云端暂时没有 GPU，也可以 CPU 跑，但会慢：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --fast \
  --device cpu
```

每个病例会输出：

```text
outputs/2026-07_totalseg_shoulder_bones/<case>/segmentations/
outputs/2026-07_totalseg_shoulder_bones/<case>/shoulder_bones_combined.nii.gz
outputs/2026-07_totalseg_shoulder_bones/<case>/run_report.json
```

其中 `shoulder_bones_combined.nii.gz` 是合并后的肩部骨 mask，包含：

- 左右肱骨
- 左右肩胛骨
- 左右锁骨

## 七、用 TotalSegmentator 骨 mask 替换形态学骨分割

生成骨 mask 后，运行传统定位方法，但指定 `--bone-mask-dir`：

```bash
python scripts/run_multibone_locator.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_bone_backend_locator \
  --bone-mask-dir outputs/2026-07_totalseg_shoulder_bones \
  --surface-arc-enable \
  --bone-edge-enable \
  --selection-policy generalized \
  --export-candidates
```

如果部分病例没有 TotalSegmentator 输出，但仍想用阈值骨分割兜底：

```bash
python scripts/run_multibone_locator.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_bone_backend_locator \
  --bone-mask-dir outputs/2026-07_totalseg_shoulder_bones \
  --allow-threshold-bone-fallback \
  --surface-arc-enable \
  --bone-edge-enable \
  --selection-policy generalized \
  --export-candidates
```

无标注 10 例仍然使用 DICOM 推理入口，默认读取：

```text
Data/unlabel
```

运行示例：

```bash
python scripts/run_multibone_dicom_inference.py \
  --data-dir Data/unlabel \
  --output-dir outputs/2026-07_unlabeled_inference \
  --surface-arc-enable \
  --bone-edge-enable \
  --selection-policy generalized \
  --export-candidates
```

## 八、查看结果

主要结果文件：

```text
outputs/2026-07_totalseg_bone_backend_locator/results/per_case_final.csv
outputs/2026-07_totalseg_bone_backend_locator/results/candidates.csv
outputs/2026-07_totalseg_bone_backend_locator/reports/summary.md
outputs/2026-07_totalseg_bone_backend_locator/previews/
```

重点比较：

- `center_error_mm`
- `doctor_roi_coverage`
- `bbox_iou`
- `bone_overlap`
- `selected_method`
- `bone_mask_source`

如果 `bone_mask_source` 显示为：

```text
outputs/2026-07_totalseg_shoulder_bones/<case>/shoulder_bones_combined.nii.gz
```

说明该病例确实使用了 TotalSegmentator 骨 mask。

如果显示为：

```text
threshold
```

说明该病例仍然使用原来的 CT 阈值骨分割。

## 九、云端结果下载

实验完成后，建议打包下载：

```bash
zip -r outputs_totalseg_bone_backend_locator.zip outputs/2026-07_totalseg_bone_backend_locator
zip -r outputs_totalseg_shoulder_bones.zip outputs/2026-07_totalseg_shoulder_bones
```

下载到本地后，可以继续让我帮你做：

- 新旧方法指标对比；
- 逐例失败分析；
- 预览图拼图；
- 给老师看的实验报告；
- 下一轮方法优化。

## 十、建议的实验顺序

1. 先只跑 1 例，例如 `LHY`，确认 TotalSegmentator 能正常输出骨 mask。
2. 再跑旧 10 例，比较使用阈值骨分割和 TotalSegmentator 骨分割后的指标。
3. 再跑新 10 例无标注数据，输出预览图做视觉质控。
4. 如果 TotalSegmentator 骨 mask 明显更稳定，再把它设为云端主实验方案。
5. 如果它在肩部局部 CT 上偶发失败，则保留阈值骨分割作为 fallback，不要完全删除原方法。

## 结论

TotalSegmentator 应作为本项目的“深度学习骨分割后端”，用于替换传统方法里的骨 mask 来源；传统候选生成、排序、质控和报告框架继续保留。这样既能利用深度学习增强骨结构识别，又不会把当前已经调好的冈上肌腱定位逻辑全部推翻。
