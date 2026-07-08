# Change Log

## 2026-06-23

- 初始化项目工程结构：`src/`、`scripts/`、`requirements/`、`configs/`、`tests/`、`docs/`。
- 新增 `README.md`，说明项目目标、文件架构、快速开始和阶段一方法路线。
- 新增 `.gitignore`，忽略原始数据、实验输出、模型权重、缓存和本地 docs。
- 新增基础数据读取、查看器、规则定位、深度学习骨架和测试文件。
- 用意：先形成可运行的 LHY 数据查看与 CT ROI 定位 baseline，同时为云端训练保留扩展接口。

### 验证记录

- `python -m unittest discover -s tests`：通过，4 个测试 OK。
- `python scripts/scan_lhy_dataset.py --root LHY\LHY --out outputs\summaries\lhy_summary.json`：通过，识别 8 个 NIfTI 和 9 个 DICOM series。
- `python scripts/export_case_summary.py --image LHY\LHY\CT\60kev.nii.gz --mask LHY\LHY\CT\ROI.nii.gz --out outputs\summaries\case_60kev_summary.json`：通过。
- `python scripts/run_rule_based_locator.py --image LHY\LHY\CT\60kev.nii.gz --target-mask LHY\LHY\CT\ROI.nii.gz --out-dir outputs\localization\lhy_60kev`：通过，输出 `roi_mask.nii.gz`、`roi.json`、`preview.png`。
- LHY 60kev 规则定位结果：`roi_recall=1.0`，`bbox_iou=0.10945401717574799`，搜索空间缩小约 `356.73` 倍。
- `python -m compileall -q src scripts`：通过。
- `python scripts/prepare_prior_channels.py --image LHY\LHY\CT\60kev.nii.gz --out-dir outputs\priors\lhy_60kev`：通过，输出 5 个路线 C 先验通道。
- `python scripts/prepare_centerline_heatmap.py --image LHY\LHY\CT\60kev.nii.gz --roi-json outputs\localization\lhy_60kev\roi.json --out outputs\localization\lhy_60kev\centerline_heatmap.nii.gz`：通过。
- `python scripts/register_mri_label_to_ct.py --help`：通过，SimpleITK 配准入口可用；实际配准需云端或已安装 SimpleITK 的环境。

## 2026-06-24

- 新增传统多骨锚点定位器 `src/supraspinatus_locator/localization/multi_bone_traditional.py`。
- 新增入口 `scripts/run_multibone_locator.py`：沿用老师 CT-only、`60keV > 300HU`、采样 ROI 半尺寸 `(22, 8, 2)` 和医生 ROI 评价标准。
- 新增对比脚本 `scripts/compare_teacher_multibone.py`，用于并排比较老师单锚点方法与多骨方法。
- 多骨方法当前使用两个锚点：近端肱骨上外侧锚点 + 肩峰/肩胛顶板骨组件；候选 ROI 放在肩峰下通道内，并继续执行骨重叠、体内比例、软组织 CT 值硬过滤。
- LHY 单例验证：老师方法 `center_error_mm=5.59`、`bbox_iou=0.0908`、`doctor_roi_coverage=0.1432`、`bone_overlap=0.0`；多骨方法 `center_error_mm=3.72`、`bbox_iou=0.1127`、`doctor_roi_coverage=0.1844`、`bone_overlap=0.0016`。

### 10 例完整验证

- 数据已归一化到 `outputs/normalized_10cases`，用于兼容老师脚本要求的 `case/CT/60keV + roi` 结构。
- 老师单锚点 baseline 输出：`outputs/teacher_10cases/evaluation/ct_tendon_locator_results.csv`。
- 多骨锚点方法输出：`outputs/multibone_10cases/multibone_locator_results.csv`。
- 对比表：`outputs/multibone_10cases/teacher_vs_multibone.csv`。
- 10 例均值：中心误差 `7.771 -> 6.311 mm`，bbox IoU `0.0578 -> 0.1089`，医生 ROI coverage `0.0851 -> 0.1855`。
- 胜负数量：中心误差 7/10 改善，bbox IoU 6/10 改善，医生 ROI coverage 7/10 改善。
- 风险：骨重叠均值 `0.0006 -> 0.0029`，仍低于老师硬约束 `0.012`，但比老师方法更高；OSQ、ZJ、SB 是主要退步/未覆盖病例。
- 新增总技术报告：`docs/ct_tendon_locator_technical_report_2026-06-24.md`，总结方法、指标、10 例结果和下一步计划。
- 新增逐例复盘：`docs/per_case_review_multibone_2026-06-24.md`，分析多骨方法骨重叠升高原因、逐例成败和后续优化优先级。

### 关键调整

- 初始“骨云中心偏移”规则在 LHY 上无重叠，已改为默认使用 `bone_bbox_fraction` 策略。
- 当前默认分数先验为 `[0.84, 0.10, 0.52]`，可通过 `--bone-bbox-fraction` 调参。
- 补充路线 C 先验通道生成、路线 D CT-MRI 弱标签配准骨架、路线 E 中心线热图生成脚本。
# 2026-06-24 多骨方法优化复盘与融合实验

- 在 `src/supraspinatus_locator/localization/multi_bone_traditional.py` 中新增 `locate_multibone_candidates(...)`，支持输出多骨候选列表。
- 重新运行 10 例多骨定位，新增 `outputs/multibone_10cases/multibone_topk_candidates.csv`，用于逐例分析 top-k 候选。
- 新增 `scripts/fuse_teacher_multibone.py`，实现老师单肱骨锚点与多骨锚点的传统规则融合。
- 生成 `outputs/hybrid_10cases/hybrid_teacher_multibone_results.csv` 与 summary。
- 新增 `docs/multibone_optimization_review_2026-06-24.md`，记录骨重叠增加原因、逐例复盘、Top-K 诊断和融合实验。
- 当前 10 例均值：老师中心误差 `7.771mm`，多骨 `6.311mm`，融合 `5.954mm`；老师 coverage `0.0851`，多骨 `0.1855`，融合 `0.2075`。

# 2026-06-24 骨距离 margin 与 z 层连续性实验

- 在 `multi_bone_traditional.py` 中新增 ROI 外扩壳层骨比例统计：`near_bone_fraction` 与 `margin_bone_fraction`。
- 在候选评分中加入骨距离软惩罚，降低“贴骨但未重叠”的候选排名。
- 新增 `anchor_continuity_score(...)`，要求候选锚点在相邻 z 层具有结构支持。
- 在 `scripts/run_multibone_locator.py` 暴露 `--bone-margin-voxels`、`--continuity-window`、`--continuity-xy-tolerance`。
- 生成 `outputs/multibone_10cases_margin_continuity/` 与 `outputs/hybrid_10cases_margin_continuity/`。
- 新增 `docs/margin_continuity_experiment_2026-06-24.md`。
- 当前最好结果为新融合：中心误差 `5.776mm`，bbox IoU `0.1248`，coverage `0.2197`，骨重叠 `0.0015`。

# 2026-06-24 老师汇报包

- 新增 `scripts/package_experiment_0624.py`，用于生成 6.24 实验汇报包。
- 在 `outputs/6.24/` 下生成老师版实验报告、结果 CSV、10 例三联对比预览图和总览图。
- 预览图颜色约定：红色为医生 ROI，黄色为老师方法，青色为本方法。
- 生成压缩包 `outputs/6.24/ct_tendon_locator_experiment_2026-06-24.zip`，便于提交或转发。

# 2026-06-29 下一轮传统多骨优化实验

- 在 `multi_bone_traditional.py` 中新增 dx/dy/dz 方向误差分解、candidate source、top-k 指标和 failure type。
- 新增 low-z exploration 与 teacher-z-refine 候选分支，候选来源包括 `current_multibone`、`low_z`、`teacher_z_refine`、`teacher_baseline`。
- 将骨 margin 评分升级为骨表面距离带评分，避免“离骨越远越好”的错误倾向。
- 将融合策略从固定 score 阈值升级为多因素规则：骨风险、teacher distance、margin shift、source confidence 等共同决策。
- 新增 `scripts/grid_search_multibone_locator.py`，支持参数敏感性实验。
- 新增 `scripts/package_next_round_experiment.py`，生成 `outputs/multibone_next_round/reports/experiment_report_next_round.md`、总览图和重点病例 top5 图。
- 当前 next_round 结果：mean center error `5.776mm`，bbox IoU `0.1581`，coverage `0.2497`，bone overlap `0.0024`，worst-case `10.63mm`。

# 2026-07-01 精细调参与打包

- 将 `scripts/grid_search_multibone_locator.py` 从默认全排列改为默认 8 个可解释 profile，避免不合理的几十万组合穷举。
- 新增 `scripts/tune_selection_from_candidates.py`，基于已生成 top-k 候选做离线融合规则坐标搜索，不重新跑 CT 候选生成。
- 新增 `scripts/package_tuned_experiment.py`，生成 `outputs/2026-07-01_tuned_experiment/`。
- 离线融合调参发现：更保守规则可降低 bone overlap 到 `0.0013`，但 coverage 会降到 `0.2324`；综合推荐仍使用 next_round final，coverage `0.2497`，bone overlap `0.0024`。
- 输出压缩包 `outputs/2026-07-01_tuned_experiment/tuned_experiment_2026-07-01.zip`。
# 2026-07-01 contact-z 方法优化与最终打包

- 在 `src/supraspinatus_locator/localization/multi_bone_traditional.py` 中新增候选生成参数：`current_anchor_count`、`branch_anchor_count`、`teacher_low_z`、`wide_xy`、`contact_z` 等，便于把“候选生成”和“候选选择”分开实验。
- 新增 `--cases` 参数，可只跑指定病例，便于快速复盘 SB/WQX/ZJ/OSQ 等问题例。
- 复盘 SB 后发现：医生 ROI 中心附近骨体素比例约 8.6%，旧规则 ROI 内骨比例上限为 1.2%，导致真实止点附近候选被系统性过滤。
- 新增 `contact_z` 分支：在老师 z 往下的中间层允许有限骨接触候选，并通过 `bone_overlap >= 0.04`、`bone_overlap <= 0.12`、低 z 补偿等门槛限制误选。
- 保留 `teacher_low_z` 与 `wide_xy` 作为候选诊断来源，但默认不参与最终自动选择；WQX 虽有轻微改善候选，但目前缺少不会误伤 LWL 的无监督选择门槛。
- 最终输出 `outputs/multibone_contact_final_v2/`：mean center error `4.903mm`，mean bbox IoU `0.1933`，mean coverage `0.3182`，mean bone overlap `0.0072`，worst-case `9.61mm`。
- 相比上一版 next_round：中心误差 `5.776 -> 4.903mm`，coverage `0.2497 -> 0.3182`，worst-case `10.63 -> 9.61mm`；代价是 bone overlap `0.0024 -> 0.0072`，主要来自 SB 的止点骨接触。
- 新增 `scripts/package_contact_final_experiment.py`，用于生成最终报告包 `outputs/2026-07-01_contact_final_experiment/`。

# 2026-07-01 surface-arc 弧形通道候选优化

- 针对医生反馈的“篮筐中心偏高偏外，肌腱沿肱骨头弧形包绕”问题，在 `multi_bone_traditional.py` 中新增 `surface_arc` 候选源。
- 新增肱骨头近似圆拟合函数 `estimate_humeral_head_circle(...)`，在肱骨头上外侧弧面生成贴骨表面候选。
- 将 `surface_arc` 的骨惩罚从“越少越好”改成分层逻辑：少量贴骨可作为贴近肱骨头表面的证据，大量深入骨内仍惩罚。
- 加入受限接管门槛：弧形候选必须是局部、有限骨接触、z层不乱跳、相对当前框沿弧面向下贴近，避免对 LHY/OSQ/ZJY 等病例误伤。
- 全量 10 例输出 `outputs/surface_arc_final/`：mean center error `3.882mm`，mean bbox IoU `0.2164`，mean coverage `0.3880`，mean bone overlap `0.0161`，worst-case `6.04mm`。
- 相比上一版 contact final：中心误差 `4.903 -> 3.882mm`，coverage `0.3182 -> 0.3880`，worst-case `9.61 -> 6.04mm`；bone overlap `0.0072 -> 0.0161`，证明骨重叠增加并不必然代表定位变差。
- 新增 `scripts/package_surface_arc_experiment.py`，用于生成 `outputs/2026-07-01_surface_arc_experiment/` 报告包。

# 2026-07-01 最终最佳版：球面平滑/中心线ROI尝试后定版

- 继续尝试肱骨头圆/球拟合稳定化：新增 `surface_arc_sphere_blend` 和 `surface_arc_sphere_anchor_count`，可用病例级中位圆心/半径平滑每层圆弧估计。
- 继续尝试多点/中心线 ROI：新增 `surface_arc_centerline_enable`、`surface_arc_centerline_points`、`surface_arc_centerline_half_size`，可用沿弧面排列的小框并集作为 surface_arc ROI。
- 验证结论：球面平滑对 WQX 略有改善，但 LWL 明显退化；中心线小框可运行，但当前 ROI 尺度下 coverage 下降。因此最终最佳默认禁用球面平滑和中心线输出，保留单框弧形候选 + 表面距离/有限贴骨评分。
- 重新运行最终最佳参数到 `outputs/surface_arc_best_final/`，指标与上一版最佳一致：mean center error `3.882mm`，mean bbox IoU `0.2164`，mean coverage `0.3880`，mean bone overlap `0.0161`，worst-case `6.04mm`。
- 新增 `scripts/package_best_final_experiment.py`，生成最终打包目录 `outputs/2026-07-01_best_final_experiment/`。
