# 2026-07-10 TotalSegmentator 引导的 HU 骨后端修正

## 问题解释

云端运行定位时出现：

```text
LWL: candidate generation failed with external bone mask (...); retrying HU threshold.
OSQ: candidate generation failed with external bone mask (...); retrying HU threshold.
WQX: candidate generation failed with external bone mask (...); retrying HU threshold.
```

这不表示 CT 数据失败，也不表示 TotalSegmentator 完全失败。更准确地说，是“TotalSegmentator 的合并骨 mask 不能直接替换旧方法里的 HU 骨阈值 mask”。

旧的多骨锚点方法依赖的是：

- CT 中 `HU > 300` 的骨皮质/高密度骨边缘；
- 2D 连通域形态；
- 肱骨头、肩峰/关节盂附近骨性边缘之间的空间关系。

TotalSegmentator 输出的是语义分割后的整块骨结构，通常是实心骨体。它适合做“这是什么骨”的语义定位，但直接替代 HU 阈值骨皮质后，连通域形态、骨边缘、骨间隙都会改变，因此旧锚点逻辑容易找不到可用候选。

## 修正策略

新增外部骨后端模式：

```text
--external-bone-mode threshold_roi
```

这是默认模式。含义是：

1. 先读取 TotalSegmentator 的肩部骨 mask。
2. 对该 mask 做轻微膨胀，默认 `--external-bone-dilation-voxels 4`。
3. 在这个空间范围内重新提取 CT 中 `HU > 300` 的骨体素。
4. 将这个 “TotalSeg-guided HU bone mask” 传给旧多骨锚点逻辑。

这样 TotalSegmentator 负责提供肩部骨的大致空间先验，HU 阈值继续保留骨皮质形态。

## 推荐命令

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

如需显式写出模式：

```bash
--external-bone-mode threshold_roi \
--external-bone-dilation-voxels 4
```

如果要复现旧的直接替换行为，可使用：

```bash
--external-bone-mode direct
```

但当前不推荐直接模式，因为它和旧锚点方法的形态假设不匹配。

## 预期现象

修正后：

- `combined=0` 或过小的病例仍会回退到纯 HU 阈值。
- TotalSegmentator 输出正常的病例会优先使用 “TotalSeg 引导 + HU 骨皮质”。
- 如果引导后的 HU 骨 mask 仍然太少，会继续回退到纯 HU 阈值，避免整批实验中断。

