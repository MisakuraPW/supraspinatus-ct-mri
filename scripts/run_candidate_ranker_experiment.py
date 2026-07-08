from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NUMERIC_FEATURES = [
    "total_score",
    "anatomical_score",
    "soft_tissue_score",
    "safety_score",
    "continuity_score",
    "z_continuity_score",
    "same_anchor_support_count",
    "rank",
    "global_rank",
    "source_rank",
    "bone_overlap",
    "near_bone_fraction",
    "margin_bone_fraction",
    "bone_distance_mm",
    "bone_distance_band_score",
    "body_inside_fraction",
    "center_y_minus_humerus_top",
    "humerus_anchor_width",
    "humerus_anchor_height",
    "arc_angle_deg",
    "surface_offset_voxels",
    "radius_normalized_distance",
    "teacher_distance_mm",
    "edge_angle_deg",
    "surface_distance_mm",
    "cortical_edge_support",
    "soft_tissue_band_mean",
    "soft_tissue_band_std",
    "bone_edge_continuity_score",
    "arc_fit_residual",
    "bone_edge_tendon_score",
]

SOURCE_FEATURES = [
    "current_multibone",
    "surface_arc",
    "bone_edge_tendon",
    "low_z",
    "contact_z",
    "teacher_z_refine",
    "teacher_low_z",
    "wide_xy",
]

NEGATIVE_VISUAL_LABELS = {
    "too_high_outer",
    "wrong_bone",
    "wrong_bone_edge",
    "too_medial_inner",
    "too_far_from_humeral_head",
    "inside_bone",
}
POSITIVE_VISUAL_LABELS = {"good", "acceptable"}


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_union(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def b(row: dict[str, object], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def group_by_case(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["case"])].append(row)
    return dict(groups)


def hard_qc_pass(row: dict[str, object]) -> bool:
    return f(row, "body_inside_fraction", 1.0) >= 0.94 and f(row, "bone_overlap", f(row, "pred_bone_overlap", 0.0)) <= 0.12


def band_score(value: float, good_min: float, good_max: float, far: float) -> float:
    if good_min <= value <= good_max:
        return 1.0
    if value < good_min:
        return max(0.0, value / max(good_min, 1e-6))
    return max(0.0, 1.0 - (value - good_max) / max(far - good_max, 1e-6))


def unified_static_score(row: dict[str, object]) -> float:
    source = str(row.get("candidate_source", ""))
    source_prior = {
        "current_multibone": 0.32,
        "contact_z": 0.40,
        "surface_arc": 0.16,
        "bone_edge_tendon": 0.14,
        "low_z": 0.12,
        "teacher_z_refine": 0.10,
        "teacher_low_z": 0.08,
        "wide_xy": 0.02,
    }.get(source, 0.0)
    bone_overlap = f(row, "bone_overlap", f(row, "pred_bone_overlap", 0.0))
    bone_distance = f(row, "bone_distance_mm", 6.0)
    y_gap = f(row, "center_y_minus_humerus_top", 8.0)
    radius_dist = f(row, "radius_normalized_distance", 1.0)
    score = (
        0.52 * f(row, "total_score")
        + 0.35 * f(row, "anatomical_score")
        + 0.20 * f(row, "soft_tissue_score")
        + 0.46 * f(row, "z_continuity_score", f(row, "continuity_score"))
        + 0.30 * min(1.0, f(row, "same_anchor_support_count") / 2.0)
        + 0.38 * band_score(bone_distance, 2.0, 8.5, 18.0)
        + 0.22 * math.exp(-((radius_dist - 1.0) ** 2) / 0.18)
        + source_prior
    )
    if source == "surface_arc" and -2.0 <= y_gap <= 26.0 and 0.006 <= bone_overlap <= 0.085:
        score += 0.18
    if source == "bone_edge_tendon":
        edge_score = f(row, "cortical_edge_support")
        residual = f(row, "arc_fit_residual", 0.25)
        surface_distance = f(row, "surface_distance_mm", bone_distance)
        score += 0.22 * edge_score
        score += 0.18 * band_score(surface_distance, 0.5, 8.0, 18.0)
        score -= 0.35 * min(1.0, residual / 0.35)
    if source == "contact_z" and 0.02 <= bone_overlap <= 0.12:
        score += 0.14
    if bone_overlap > 0.085:
        score -= 1.60 * (bone_overlap - 0.085) / 0.035
    if f(row, "near_bone_fraction") > 0.16:
        score -= 0.45
    if f(row, "margin_bone_fraction") > 0.18:
        score -= 0.32
    if y_gap < -8.0:
        score -= 0.42
    if y_gap > 36.0:
        score -= 0.36
    if b(row, "detached_from_bone"):
        score -= 0.85
    if b(row, "suspicious_narrow_anchor"):
        score -= 0.38
    if b(row, "possible_wrong_bone"):
        score -= 0.62
    if not hard_qc_pass(row):
        score -= 10.0
    return score


def weak_surface_contact(row: dict[str, object]) -> bool:
    bone_overlap = f(row, "bone_overlap", f(row, "pred_bone_overlap", 0.0))
    near_bone = f(row, "near_bone_fraction")
    radius_dist = f(row, "radius_normalized_distance", 1.0)
    return (bone_overlap <= 0.004 and near_bone <= 0.03) or (
        bone_overlap <= 0.010 and near_bone <= 0.035 and radius_dist >= 1.45
    )


def stable_surface_channel_candidate(row: dict[str, object]) -> bool:
    source = str(row.get("candidate_source", ""))
    if source not in {"contact_z", "surface_arc", "bone_edge_tendon"}:
        return False
    bone_overlap = f(row, "bone_overlap", f(row, "pred_bone_overlap", 0.0))
    near_bone = f(row, "near_bone_fraction")
    if not (0.020 <= bone_overlap <= 0.120 and 0.040 <= near_bone <= 0.150):
        return False
    if f(row, "margin_bone_fraction") > 0.140:
        return False
    if f(row, "z_continuity_score", f(row, "continuity_score")) < 0.600:
        return False
    min_total = {"contact_z": 2.35, "surface_arc": 4.00, "bone_edge_tendon": 2.80}[source]
    return f(row, "total_score") >= min_total


def surface_channel_priority(row: dict[str, object], score_key: str) -> float:
    source = str(row.get("candidate_source", ""))
    source_bonus = {"contact_z": 0.28, "surface_arc": 0.18, "bone_edge_tendon": 0.22}.get(source, 0.0)
    bone_overlap = f(row, "bone_overlap", f(row, "pred_bone_overlap", 0.0))
    near_bone = f(row, "near_bone_fraction")
    z_score = f(row, "z_continuity_score", f(row, "continuity_score"))
    edge_score = f(row, "bone_edge_continuity_score", z_score)
    return (
        f(row, score_key)
        + 0.10 * f(row, "total_score")
        + 0.16 * band_score(bone_overlap, 0.035, 0.085, 0.130)
        + 0.10 * band_score(near_bone, 0.055, 0.120, 0.170)
        + 0.12 * z_score
        + 0.08 * edge_score
        + source_bonus
    )


def select_with_scores(rows: list[dict[str, object]], score_key: str = "unified_score") -> dict[str, object]:
    qc_rows = [row for row in rows if hard_qc_pass(row)]
    pool = qc_rows or rows
    ranked = sorted(pool, key=lambda row: f(row, score_key), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    gap = f(top, score_key) - (f(second, score_key) if second else -999.0)
    force_review = False
    contact_like = [
        row
        for row in pool
        if row.get("candidate_source") == "contact_z"
        and 0.04 <= f(row, "bone_overlap", f(row, "pred_bone_overlap")) <= 0.09
        and 0.055 <= f(row, "near_bone_fraction") <= 0.14
        and f(row, "margin_bone_fraction") <= 0.14
        and f(row, "total_score") >= 2.35
    ]
    contact_like.sort(key=lambda row: (f(row, score_key), f(row, "total_score")), reverse=True)
    if contact_like and (
        (f(top, "bone_overlap", f(top, "pred_bone_overlap")) <= 0.004 and f(top, "near_bone_fraction") <= 0.025)
        or str(top.get("candidate_source")) == "surface_arc"
    ):
        top = contact_like[0]
        second = ranked[0] if ranked[0] is not top else (ranked[1] if len(ranked) > 1 else None)
        gap = f(top, score_key) - (f(second, score_key) if second else -999.0)
        top["rescue_reason"] = "contact_like_anatomic_review"

    surface_channel = [row for row in pool if stable_surface_channel_candidate(row)]
    surface_channel.sort(key=lambda row: surface_channel_priority(row, score_key), reverse=True)
    if surface_channel and weak_surface_contact(top):
        top["review_alternative_candidate_id"] = surface_channel[0].get("candidate_id", "")
        top["review_alternative_source"] = surface_channel[0].get("candidate_source", "")
        top["review_alternative_score"] = round(float(surface_channel_priority(surface_channel[0], score_key)), 4)
        top["review_reason"] = "stable_surface_channel_alternative"
        force_review = True

    if str(top.get("candidate_source")) == "surface_arc" and f(top, "margin_bone_fraction") > 0.095:
        non_surface = [row for row in ranked if row.get("candidate_source") != "surface_arc"]
        if non_surface:
            top = non_surface[0]
            second = ranked[0] if ranked[0] is not top else (ranked[1] if len(ranked) > 1 else None)
            gap = f(top, score_key) - (f(second, score_key) if second else -999.0)
            top["rescue_reason"] = "surface_margin_too_deep_review"

    top["review_status"] = "needs_review" if force_review or (second is not None and gap < 0.20) else "auto_select"
    top["score_gap_to_second"] = round(float(gap), 4)
    return top


def metrics_from_final_rows(policy: str, rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"policy": policy, "cases": 0}
    errors = np.asarray([f(row, "center_error_mm") for row in rows], dtype=float)
    ious = np.asarray([f(row, "pred_box_doctor_bbox_iou", f(row, "bbox_iou")) for row in rows], dtype=float)
    coverage = np.asarray([f(row, "doctor_roi_coverage") for row in rows], dtype=float)
    bone = np.asarray([f(row, "pred_bone_overlap", f(row, "bone_overlap")) for row in rows], dtype=float)
    return {
        "policy": policy,
        "cases": len(rows),
        "mean_center_error_mm": round(float(errors.mean()), 3),
        "median_center_error_mm": round(float(np.median(errors)), 3),
        "worst_center_error_mm": round(float(errors.max()), 3),
        "mean_bbox_iou": round(float(ious.mean()), 4),
        "mean_doctor_roi_coverage": round(float(coverage.mean()), 4),
        "mean_pred_bone_overlap": round(float(bone.mean()), 4),
        "needs_review_count": sum(1 for row in rows if row.get("review_status") == "needs_review"),
        "selected_source_counts": "; ".join(f"{k}:{v}" for k, v in sorted(Counter(str(row.get("selected_method", row.get("candidate_source", ""))) for row in rows).items())),
    }


def candidate_label_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    labeled: list[dict[str, object]] = []
    for case, case_rows in group_by_case(rows).items():
        best_error = min(f(row, "center_error_mm", 999.0) for row in case_rows)
        best_cov = max(f(row, "doctor_roi_coverage") for row in case_rows)
        for row in case_rows:
            row = dict(row)
            positive = (
                f(row, "center_error_mm", 999.0) <= best_error + 1.5
                or (
                    f(row, "center_error_mm", 999.0) <= 6.0
                    and best_cov >= 0.10
                    and f(row, "doctor_roi_coverage") >= max(0.12, best_cov - 0.08)
                )
            )
            row["ranker_label"] = 1 if positive else 0
            labeled.append(row)
    return labeled


def feature_vector(row: dict[str, object]) -> list[float]:
    values = [f(row, key) for key in NUMERIC_FEATURES]
    source = str(row.get("candidate_source", ""))
    values.extend(1.0 if source == feature_source else 0.0 for feature_source in SOURCE_FEATURES)
    values.extend(
        [
            1.0 if b(row, "detached_from_bone") else 0.0,
            1.0 if b(row, "suspicious_narrow_anchor") else 0.0,
            1.0 if b(row, "possible_wrong_bone") else 0.0,
            unified_static_score(row),
        ]
    )
    return values


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def train_logistic(train_x: np.ndarray, train_y: np.ndarray, epochs: int = 800, lr: float = 0.06) -> np.ndarray:
    weights = np.zeros(train_x.shape[1] + 1, dtype=float)
    x_aug = np.c_[np.ones(train_x.shape[0]), train_x]
    pos_weight = len(train_y) / max(1.0, 2.0 * float(train_y.sum()))
    neg_weight = len(train_y) / max(1.0, 2.0 * float((1 - train_y).sum()))
    sample_weight = np.where(train_y > 0.5, pos_weight, neg_weight)
    for _ in range(epochs):
        logits = np.clip(x_aug @ weights, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = x_aug.T @ ((pred - train_y) * sample_weight) / len(train_y)
        weights -= lr * grad
    return weights


def predict_logistic(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.c_[np.ones(x.shape[0]), x]
    logits = np.clip(x_aug @ weights, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def candidate_utility(row: dict[str, object]) -> float:
    error = f(row, "center_error_mm", 999.0)
    coverage = f(row, "doctor_roi_coverage")
    iou = f(row, "pred_box_doctor_bbox_iou", f(row, "bbox_iou"))
    return -error + 3.0 * coverage + 1.5 * iou


def case_pair_indices(case_rows: list[dict[str, object]], min_utility_gap: float = 1.0, max_pairs: int = 3000) -> list[tuple[int, int]]:
    utilities = [candidate_utility(row) for row in case_rows]
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(case_rows)):
        for j in range(i + 1, len(case_rows)):
            gap = utilities[i] - utilities[j]
            if abs(gap) < min_utility_gap:
                continue
            winner, loser = (i, j) if gap > 0 else (j, i)
            pairs.append((abs(gap), winner, loser))
    pairs.sort(reverse=True)
    if len(pairs) <= max_pairs:
        return [(winner, loser) for _, winner, loser in pairs]
    stride = max(1, len(pairs) // max_pairs)
    sampled = pairs[::stride][:max_pairs]
    return [(winner, loser) for _, winner, loser in sampled]


def pairwise_training_matrix(rows: list[dict[str, object]], feature_lookup: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    diffs = []
    labels = []
    case_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        case_indices[str(row["case"])].append(idx)
    for indices in case_indices.values():
        case_only_rows = [rows[idx] for idx in indices]
        for winner_local, loser_local in case_pair_indices(case_only_rows):
            winner_idx = indices[winner_local]
            loser_idx = indices[loser_local]
            diff = feature_lookup[winner_idx] - feature_lookup[loser_idx]
            diffs.append(diff)
            labels.append(1.0)
            diffs.append(-diff)
            labels.append(0.0)
    if not diffs:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)
    return np.asarray(diffs, dtype=float), np.asarray(labels, dtype=float)


def train_pairwise_predict(training_rows: list[dict[str, object]], target_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not training_rows or not target_rows:
        return target_rows
    train_x_raw = np.asarray([feature_vector(row) for row in training_rows], dtype=float)
    target_x_raw = np.asarray([feature_vector(row) for row in target_rows], dtype=float)
    train_x, target_x = standardize(train_x_raw, target_x_raw)
    train_feature_lookup = {idx: train_x[idx] for idx in range(len(training_rows))}
    pair_x, pair_y = pairwise_training_matrix(training_rows, train_feature_lookup)
    if pair_x.size == 0 or len({int(y) for y in pair_y}) < 2:
        out = []
        for row in target_rows:
            row = dict(row)
            row["pairwise_score"] = unified_static_score(row)
            out.append(row)
        return out
    weights = train_logistic(pair_x, pair_y, epochs=650, lr=0.045)
    out_rows = [dict(row) for row in target_rows]
    target_groups = group_by_case(out_rows)
    row_to_index = {id(row): idx for idx, row in enumerate(out_rows)}
    for case_rows in target_groups.values():
        indices = [row_to_index[id(row)] for row in case_rows]
        if len(indices) == 1:
            out_rows[indices[0]]["pairwise_score"] = 1.0
            continue
        case_x = target_x[indices]
        static_scores = np.asarray([unified_static_score(out_rows[idx]) for idx in indices], dtype=float)
        static_scores = (static_scores - static_scores.min()) / max(1e-6, static_scores.max() - static_scores.min())
        for local_idx, row_idx in enumerate(indices):
            diffs = case_x[local_idx] - np.delete(case_x, local_idx, axis=0)
            probs = predict_logistic(weights, diffs)
            static = static_scores[local_idx]
            out_rows[row_idx]["pairwise_score"] = round(float(0.78 * probs.mean() + 0.22 * static), 5)
    return out_rows


def loocv_pairwise_predictions(labeled_candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    cases = sorted(group_by_case(labeled_candidates))
    predictions: list[dict[str, object]] = []
    for held_case in cases:
        train_rows = [dict(row) for row in labeled_candidates if str(row["case"]) != held_case]
        test_rows = [dict(row) for row in labeled_candidates if str(row["case"]) == held_case]
        scored = train_pairwise_predict(train_rows, test_rows)
        selected = dict(select_with_scores(scored, score_key="pairwise_score"))
        selected["policy"] = "loocv_pairwise_ranker"
        selected["selected_method"] = selected.get("candidate_source", "")
        predictions.append(selected)
    return predictions, metrics_from_final_rows("loocv_pairwise_ranker", predictions)


def loocv_predictions(labeled_candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    labeled = candidate_label_rows([dict(row) for row in labeled_candidates])
    cases = sorted(group_by_case(labeled))
    predictions: list[dict[str, object]] = []
    for held_case in cases:
        train_rows = [row for row in labeled if str(row["case"]) != held_case]
        test_rows = [row for row in labeled if str(row["case"]) == held_case]
        if not train_rows or not test_rows or len({int(row["ranker_label"]) for row in train_rows}) < 2:
            for row in test_rows:
                row["learning_score"] = unified_static_score(row)
        else:
            train_x = np.asarray([feature_vector(row) for row in train_rows], dtype=float)
            train_y = np.asarray([int(row["ranker_label"]) for row in train_rows], dtype=float)
            test_x = np.asarray([feature_vector(row) for row in test_rows], dtype=float)
            train_x, test_x = standardize(train_x, test_x)
            weights = train_logistic(train_x, train_y)
            probs = predict_logistic(weights, test_x)
            static_scores = np.asarray([unified_static_score(row) for row in test_rows], dtype=float)
            static_scores = (static_scores - static_scores.min()) / max(1e-6, static_scores.max() - static_scores.min())
            for row, prob, static_score in zip(test_rows, probs, static_scores):
                row["learning_score"] = round(float(0.72 * prob + 0.28 * static_score), 5)
        selected = select_with_scores(test_rows, score_key="learning_score")
        selected = dict(selected)
        selected["policy"] = "loocv_logistic_ranker"
        selected["selected_method"] = selected.get("candidate_source", "")
        predictions.append(selected)
    return predictions, metrics_from_final_rows("loocv_logistic_ranker", predictions)


def train_from_labeled_rows(training_rows: list[dict[str, object]], target_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not training_rows or len({int(row["ranker_label"]) for row in training_rows}) < 2:
        for row in target_rows:
            row["learning_score"] = unified_static_score(row)
        return target_rows
    train_x = np.asarray([feature_vector(row) for row in training_rows], dtype=float)
    train_y = np.asarray([int(row["ranker_label"]) for row in training_rows], dtype=float)
    target_x = np.asarray([feature_vector(row) for row in target_rows], dtype=float)
    train_x, target_x = standardize(train_x, target_x)
    weights = train_logistic(train_x, train_y)
    probs = predict_logistic(weights, target_x)
    static_scores = np.asarray([unified_static_score(row) for row in target_rows], dtype=float)
    static_scores = (static_scores - static_scores.min()) / max(1e-6, static_scores.max() - static_scores.min())
    out = []
    for row, prob, static_score in zip(target_rows, probs, static_scores):
        row = dict(row)
        row["learning_score"] = round(float(0.72 * prob + 0.28 * static_score), 5)
        out.append(row)
    return out


def train_all_labeled_predict(rows: list[dict[str, object]], target_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    labeled = candidate_label_rows([dict(row) for row in rows])
    return train_from_labeled_rows(labeled, target_rows)


def attach_visual_feedback(rows: list[dict[str, object]], feedback_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    feedback = {row.get("candidate_id", ""): row for row in feedback_rows if row.get("visual_label")}
    out = []
    for row in rows:
        merged = dict(row)
        fb = feedback.get(str(row.get("candidate_id", "")))
        if fb:
            label = fb.get("visual_label", "")
            merged["visual_label"] = label
            merged["visual_note"] = fb.get("note", "")
            if label in POSITIVE_VISUAL_LABELS:
                merged["ranker_label"] = 1
            elif label in NEGATIVE_VISUAL_LABELS:
                merged["ranker_label"] = 0
        out.append(merged)
    return out


def select_static_by_case(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_rows = []
    for case, case_rows in group_by_case(rows).items():
        scored = []
        for row in case_rows:
            row = dict(row)
            row["unified_score"] = round(float(unified_static_score(row)), 5)
            scored.append(row)
        selected = dict(select_with_scores(scored, score_key="unified_score"))
        selected["policy"] = "unified_ranker_policy"
        selected["selected_method"] = selected.get("candidate_source", "")
        selected_rows.append(selected)
    return sorted(selected_rows, key=lambda row: str(row["case"]))


def select_source_only_by_case(rows: list[dict[str, object]], source: str, policy: str) -> list[dict[str, object]]:
    selected_rows = []
    for case, case_rows in group_by_case(rows).items():
        source_rows = [dict(row) for row in case_rows if row.get("candidate_source") == source]
        if not source_rows:
            continue
        source_rows.sort(key=lambda row: f(row, "total_score"), reverse=True)
        selected = source_rows[0]
        selected["policy"] = policy
        selected["selected_method"] = selected.get("candidate_source", "")
        selected_rows.append(selected)
    return sorted(selected_rows, key=lambda row: str(row["case"]))


def select_anchor_review_by_case(
    anchor_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    policy: str,
) -> list[dict[str, object]]:
    candidates_by_case = group_by_case(candidate_rows)
    selected_rows = []
    for anchor in anchor_rows:
        row = dict(anchor)
        case = str(row.get("case", ""))
        pool = candidates_by_case.get(case, [])
        row["policy"] = policy
        row["selected_method"] = row.get("selected_method", row.get("candidate_source", row.get("method", "")))
        row["review_status"] = "auto_select"
        if not pool:
            selected_rows.append(row)
            continue

        surface_channel = [candidate for candidate in pool if stable_surface_channel_candidate(candidate)]
        surface_channel.sort(key=lambda candidate: surface_channel_priority(candidate, "total_score"), reverse=True)
        if surface_channel and weak_surface_contact(row):
            alternative = surface_channel[0]
            row["review_status"] = "needs_review"
            row["review_reason"] = "anchor_weak_contact_surface_channel_alternative"
            row["review_alternative_candidate_id"] = alternative.get("candidate_id", "")
            row["review_alternative_source"] = alternative.get("candidate_source", "")
            row["review_alternative_score"] = round(float(surface_channel_priority(alternative, "total_score")), 4)
            row["review_alternative_center_error_mm"] = alternative.get("center_error_mm", "")
            row["review_alternative_coverage"] = alternative.get("doctor_roi_coverage", "")
        selected_rows.append(row)
    return sorted(selected_rows, key=lambda item: str(item.get("case", "")))


def topk_by_case(rows: list[dict[str, object]], score_key: str, k: int = 3) -> list[dict[str, object]]:
    out = []
    for case, case_rows in group_by_case(rows).items():
        ranked = sorted(case_rows, key=lambda row: f(row, score_key), reverse=True)
        for idx, row in enumerate(ranked[:k], start=1):
            item = dict(row)
            item["ranker_rank"] = idx
            out.append(item)
    return sorted(out, key=lambda row: (str(row["case"]), int(row["ranker_rank"])))


def write_report(path: Path, summary_rows: list[dict[str, object]], loocv_summary: dict[str, object], unlabeled_top3: list[dict[str, object]]) -> None:
    lines = [
        "# Candidate Bank Robust Ranker Experiment",
        "",
        "This run keeps all morphology routes as candidate generators, then applies one shared QC and ranking layer.",
        "",
        "## Labeled Metrics",
        "",
        "| policy | mean error | worst error | coverage | IoU | bone overlap | review |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('policy')} | {row.get('mean_center_error_mm', '')} | {row.get('worst_center_error_mm', '')} | "
            f"{row.get('mean_doctor_roi_coverage', '')} | {row.get('mean_bbox_iou', '')} | {row.get('mean_pred_bone_overlap', '')} | {row.get('needs_review_count', '')} |"
        )
    lines.extend(
        [
            "",
            "## Leave-One-Case-Out",
            "",
            f"- mean_center_error_mm: {loocv_summary.get('mean_center_error_mm', '')}",
            f"- worst_center_error_mm: {loocv_summary.get('worst_center_error_mm', '')}",
            f"- mean_doctor_roi_coverage: {loocv_summary.get('mean_doctor_roi_coverage', '')}",
            f"- mean_pred_bone_overlap: {loocv_summary.get('mean_pred_bone_overlap', '')}",
            "",
            "## Unlabeled Visual Set",
            "",
            "For unlabeled cases the script reports top-3 candidates instead of pretending one answer is ground truth.",
            "",
            "| case | rank | source | score | bone overlap | flags |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for row in unlabeled_top3:
        flags = ",".join(
            key
            for key in ("detached_from_bone", "suspicious_narrow_anchor", "possible_wrong_bone")
            if b(row, key)
        )
        lines.append(
            f"| {row.get('case')} | {row.get('ranker_rank')} | {row.get('candidate_source')} | "
            f"{row.get('learning_score', row.get('unified_score', ''))} | {row.get('bone_overlap', row.get('pred_bone_overlap', ''))} | {flags} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate-pool selection policies and run LOOCV ranker.")
    parser.add_argument("--labeled-candidates", default="outputs/2026-07_candidate_bank/labeled/results/per_case_topk.csv")
    parser.add_argument("--unlabeled-candidates", default="outputs/2026-07_candidate_bank/unlabeled/results/per_case_topk.csv")
    parser.add_argument("--unlabeled-feedback", default="outputs/2026-07_candidate_bank/unlabeled/results/unlabeled_visual_feedback_template.csv")
    parser.add_argument("--old-best-final", default="outputs/surface_arc_best_final/results/per_case_final.csv")
    parser.add_argument("--generalized-final", default="outputs/2026-07-02_labeled_generalized_rescue_no_teacher/results/per_case_final.csv")
    parser.add_argument("--labeled-anchor-final", default="")
    parser.add_argument("--unlabeled-anchor-final", default="")
    parser.add_argument("--output-dir", default="outputs/2026-07_candidate_ranker_experiment")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    reports_dir = output_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    labeled_candidates = [dict(row) for row in read_csv_rows(Path(args.labeled_candidates))]
    unlabeled_candidates = [dict(row) for row in read_csv_rows(Path(args.unlabeled_candidates))]
    labeled_anchor_rows = read_csv_rows(Path(args.labeled_anchor_final)) if args.labeled_anchor_final else []
    unlabeled_anchor_rows = read_csv_rows(Path(args.unlabeled_anchor_final)) if args.unlabeled_anchor_final else []
    feedback_rows = read_csv_rows(Path(args.unlabeled_feedback))
    unlabeled_candidates = attach_visual_feedback(unlabeled_candidates, feedback_rows)

    static_labeled = select_static_by_case(labeled_candidates)
    bone_edge_only_labeled = select_source_only_by_case(labeled_candidates, "bone_edge_tendon", "bone_edge_tendon_only")
    anchor_labeled = select_anchor_review_by_case(labeled_anchor_rows, labeled_candidates, "anchor_review_policy") if labeled_anchor_rows else []
    learned_labeled_rows = train_all_labeled_predict(labeled_candidates, [dict(row) for row in labeled_candidates])
    learned_labeled = []
    for case_rows in group_by_case(learned_labeled_rows).values():
        selected = dict(select_with_scores(case_rows, score_key="learning_score"))
        selected["policy"] = "unified_ranker_policy"
        selected["selected_method"] = selected.get("candidate_source", "")
        learned_labeled.append(selected)
    learned_labeled.sort(key=lambda row: str(row["case"]))
    loocv_rows, loocv_summary = loocv_predictions(labeled_candidates)
    pairwise_labeled_rows = train_pairwise_predict([dict(row) for row in labeled_candidates], [dict(row) for row in labeled_candidates])
    pairwise_labeled = []
    for case_rows in group_by_case(pairwise_labeled_rows).values():
        selected = dict(select_with_scores(case_rows, score_key="pairwise_score"))
        selected["policy"] = "pairwise_ranker_policy"
        selected["selected_method"] = selected.get("candidate_source", "")
        pairwise_labeled.append(selected)
    pairwise_labeled.sort(key=lambda row: str(row["case"]))
    pairwise_loocv_rows, pairwise_loocv_summary = loocv_pairwise_predictions(labeled_candidates)
    training_rows = candidate_label_rows([dict(row) for row in labeled_candidates])
    training_rows.extend(dict(row) for row in unlabeled_candidates if "ranker_label" in row)
    learned_unlabeled_rows = train_from_labeled_rows(training_rows, unlabeled_candidates)
    pairwise_unlabeled_rows = train_pairwise_predict([dict(row) for row in labeled_candidates], [dict(row) for row in unlabeled_candidates])
    unlabeled_top3 = topk_by_case(learned_unlabeled_rows, score_key="learning_score", k=3)
    unlabeled_selected = [select_with_scores(rows, score_key="learning_score") for rows in group_by_case(learned_unlabeled_rows).values()]
    pairwise_unlabeled_top3 = topk_by_case(pairwise_unlabeled_rows, score_key="pairwise_score", k=3)
    pairwise_unlabeled_selected = [select_with_scores(rows, score_key="pairwise_score") for rows in group_by_case(pairwise_unlabeled_rows).values()]
    anchor_unlabeled = select_anchor_review_by_case(unlabeled_anchor_rows, unlabeled_candidates, "anchor_review_policy") if unlabeled_anchor_rows else []

    old_best_rows = read_csv_rows(Path(args.old_best_final))
    generalized_rows = read_csv_rows(Path(args.generalized_final))
    summary_rows = [
        metrics_from_final_rows("old_best_policy", old_best_rows),
        metrics_from_final_rows("generalized_policy", generalized_rows),
        metrics_from_final_rows("bone_edge_tendon_only", bone_edge_only_labeled),
        metrics_from_final_rows("anchor_review_policy", anchor_labeled) if anchor_labeled else {"policy": "anchor_review_policy", "cases": 0},
        metrics_from_final_rows("unified_static_policy", static_labeled),
        metrics_from_final_rows("unified_ranker_policy", learned_labeled),
        loocv_summary,
        metrics_from_final_rows("pairwise_ranker_policy", pairwise_labeled),
        pairwise_loocv_summary,
    ]

    write_csv_union(results_dir / "policy_summary.csv", summary_rows)
    write_csv_union(results_dir / "unified_labeled_selections.csv", static_labeled)
    write_csv_union(results_dir / "bone_edge_tendon_only_labeled_selections.csv", bone_edge_only_labeled)
    write_csv_union(results_dir / "anchor_labeled_selections.csv", anchor_labeled)
    write_csv_union(results_dir / "unified_learned_labeled_selections.csv", learned_labeled)
    write_csv_union(results_dir / "loocv_predictions.csv", loocv_rows)
    write_csv_union(results_dir / "loocv_summary.csv", [loocv_summary])
    write_csv_union(results_dir / "pairwise_labeled_selections.csv", pairwise_labeled)
    write_csv_union(results_dir / "pairwise_loocv_predictions.csv", pairwise_loocv_rows)
    write_csv_union(results_dir / "pairwise_loocv_summary.csv", [pairwise_loocv_summary])
    write_csv_union(results_dir / "unlabeled_unified_top3.csv", unlabeled_top3)
    write_csv_union(results_dir / "unlabeled_unified_selections.csv", unlabeled_selected)
    write_csv_union(results_dir / "unlabeled_pairwise_top3.csv", pairwise_unlabeled_top3)
    write_csv_union(results_dir / "unlabeled_pairwise_selections.csv", pairwise_unlabeled_selected)
    write_csv_union(results_dir / "unlabeled_anchor_selections.csv", anchor_unlabeled)
    write_report(reports_dir / "ranker_experiment_report.md", summary_rows, loocv_summary, unlabeled_top3)
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
