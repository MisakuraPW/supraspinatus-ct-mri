# 2026-07-10 TotalSegmentator 云端失败诊断记录

## 背景

云端运行命令：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --fast \
  --device gpu
```

第一例 `HMC` 返回 `returncode=1`，日志位于该病例输出目录的 `stdout.txt` 和 `stderr.txt`。

## 日志结论

这次失败不是 HMC 数据读取失败，也不是本项目的肩部骨 mask 合并代码失败。TotalSegmentator 在真正推理前需要首次下载模型权重，下载 `Task 297` 时网络中断：

```text
requests.exceptions.ChunkedEncodingError
IncompleteRead(4963674 bytes read, 130422401 more expected)
```

同时云端 PyTorch 没有成功启用 GPU：

```text
No GPU detected. Running on CPU.
CUDA initialization: The NVIDIA driver on your system is too old ...
```

因此当前有两个独立问题：

1. 模型权重下载中断，只下载了约 4.96 MB / 135 MB。
2. PyTorch/CUDA/驱动版本不匹配，TotalSegmentator 退回 CPU。

## 建议云端处理顺序

先检查 GPU 环境：

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

如果 `torch.cuda.is_available()` 是 `False`，优先换成云平台提供的 PyTorch/CUDA 匹配镜像，或者重装一个不高于当前驱动支持版本的 PyTorch CUDA wheel。不要在 GPU 不可用时直接批量跑全部病例。

然后单独预下载 TotalSegmentator 权重，成功后再跑病例：

```bash
for i in 1 2 3 4 5; do
  totalseg_download_weights -t total_fast && break
  sleep 30
done
```

如果云端网络一直断，可以在网络更稳定的机器上下载 TotalSegmentator 权重目录，再复制到云端用户目录。常见位置是：

```bash
~/.totalsegmentator
```

## 推荐重新运行方式

先只跑 HMC，一例成功后再全量跑：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --cases HMC \
  --fast \
  --device gpu \
  --stop-on-fail
```

如果 GPU 仍不可用但想验证流程，可以临时 CPU 跑一例：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --cases HMC \
  --fast \
  --device cpu \
  --stop-on-fail
```

HMC 成功后再运行全量：

```bash
python scripts/run_totalseg_shoulder_bones.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_shoulder_bones \
  --fast \
  --device gpu \
  --skip-existing \
  --stop-on-fail
```

## 本次代码改动

`scripts/run_totalseg_shoulder_bones.py` 增加了：

- `failure_hint` 字段：在 CSV 汇总中记录失败类型。
- 下载中断识别：`ChunkedEncodingError`、`IncompleteRead`、`connection broken`。
- GPU 环境异常识别：`No GPU detected`、`CUDA initialization`、驱动过旧。
- `--stop-on-fail`：第一例失败时立即停止，避免批量任务继续空跑。

