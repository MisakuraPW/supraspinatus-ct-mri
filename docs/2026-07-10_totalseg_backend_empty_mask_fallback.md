# 2026-07-10 TotalSegmentator 骨后端空 mask 回退修复

## 问题

云端 TotalSegmentator 已经成功启用 GPU：

```text
torch=2.11.0+cu128 cuda_available=True gpu=NVIDIA A10
```

但肩部骨分割结果中出现了异常：

```text
HMC combined=4608
LHY combined=0
ZH combined=0
```

`combined` 是合并后的肩部骨 mask 体素数，不是评分。正常病例一般是几十万量级；`0` 表示 TotalSegmentator 对该病例没有输出可用肩部骨 mask，`4608` 也明显偏小。

后续定位器使用这些外部骨 mask 时，会因为骨结构为空或过少而无法生成候选，报错：

```text
ValueError: Could not locate a valid multi-bone supraspinatus sampling ROI
```

## 修复

`scripts/run_multibone_locator.py` 新增参数：

```text
--min-external-bone-voxels
```

默认值为 `10000`。当外部骨 mask 小于该阈值时，认为该 TotalSegmentator 输出不可靠。

`src/supraspinatus_locator/localization/multi_bone_traditional.py` 新增逻辑：

- 如果外部骨 mask 缺失，且启用了 `--allow-threshold-bone-fallback`，回退到原始 HU 阈值骨分割。
- 如果外部骨 mask 体素数过少，且启用了 `--allow-threshold-bone-fallback`，回退到原始 HU 阈值骨分割。
- 如果外部骨 mask 非空但候选生成失败，且启用了 `--allow-threshold-bone-fallback`，再次用 HU 阈值骨分割重试。
- 候选 CSV 中新增 `external_bone_voxels`，用于记录外部骨 mask 体素数。

## 推荐云端命令

```bash
python scripts/run_multibone_locator.py \
  --data-dir Data/label \
  --output-dir outputs/2026-07_totalseg_bone_backend_locator \
  --bone-mask-dir outputs/2026-07_totalseg_shoulder_bones \
  --allow-threshold-bone-fallback \
  --external-bone-mode threshold_roi \
  --surface-arc-enable \
  --bone-edge-enable \
  --selection-policy generalized \
  --export-candidates
```

如果想更严格地只接受体素更多的 TotalSegmentator 骨 mask，可以调高：

```bash
--min-external-bone-voxels 50000
```

## 注意

这个修复不是说 TotalSegmentator 的空 mask 被修好了，而是让它作为外部骨后端时不会拖垮整批实验。对于 `LHY/ZH/HMC` 这类异常病例，后续仍应单独查看 TotalSegmentator 输出，判断是输入 CT、方向、裁剪范围、fast 模型分辨率，还是 TotalSegmentator 对肩部小范围 CT 的泛化问题。

后续又新增了 `threshold_roi` 模式：TotalSegmentator 的有效 mask 默认不再直接替换 HU 骨阈值，而是作为空间先验，引导提取 CT 中的 `HU > 300` 骨皮质。这样更符合旧多骨锚点方法的形态假设。
