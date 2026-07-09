# 2026-07-10 TotalSegmentator 权重本地缓存与云端迁移

## 本地缓存位置

本机已下载 TotalSegmentator `total_fast` 权重：

```text
outputs/2026-07_totalseg_weights_cache/.totalsegmentator
```

包含两个模型目录：

```text
Dataset297_TotalSegmentator_total_3mm_1559subj
Dataset298_TotalSegmentator_total_6mm_1559subj
```

已打包为：

```text
outputs/2026-07_totalseg_weights_cache/totalseg_total_fast_weights.zip
```

## 云端使用方式

把 `totalseg_total_fast_weights.zip` 上传到云端项目目录，例如：

```text
/mnt/workspace/supraspinatus-ct-mri/totalseg_total_fast_weights.zip
```

然后在云端解压：

```bash
unzip -o totalseg_total_fast_weights.zip -d outputs/2026-07_totalseg_weights_cache
```

解压后应出现：

```text
outputs/2026-07_totalseg_weights_cache/.totalsegmentator/nnunet/results/...
```

运行 TotalSegmentator 前设置环境变量：

```bash
export TOTALSEG_HOME_DIR=/mnt/workspace/supraspinatus-ct-mri/outputs/2026-07_totalseg_weights_cache/.totalsegmentator
```

之后先跑一例验证：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --cases HMC \
  --fast \
  --device gpu \
  --stop-on-fail
```

如果 HMC 成功，再全量运行：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --fast \
  --device gpu \
  --skip-existing \
  --stop-on-fail
```

## 说明

这个 zip 只包含 TotalSegmentator 的预训练模型缓存，不包含本项目数据。它的作用是避免云端第一次运行时从外网慢速下载权重。

