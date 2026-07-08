from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


FOCUS_CASES = ("SB", "WQX", "ZJ", "OSQ")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def teacher_candidate(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    out["candidate_id"] = f"{row['case']}_teacher"
    out["candidate_source"] = "teacher_baseline"
    out["total_score"] = "0"
    out["bone_overlap"] = row.get("pred_bone_overlap", "0")
    out["bbox_iou"] = row.get("pred_box_doctor_bbox_iou", "0")
    out["teacher_distance_mm"] = "0"
    out["decision_reason"] = "teacher_baseline_candidate"
    return out


def decision_score(row: dict[str, str], params: dict[str, float]) -> float:
    score = f(row, "total_score")
    source = row.get("candidate_source", "")
    if source == "current_multibone":
        score += params["current_bonus"]
    elif source == "teacher_z_refine":
        score -= params["teacher_refine_penalty"]
    elif source == "low_z":
        score -= params["low_z_penalty"]
    elif source == "teacher_baseline":
        score += params["teacher_bonus"]
    score += f(row, "bone_distance_band_score") * params["band_weight_delta"]
    score -= f(row, "near_bone_fraction") * params["near_weight"]
    score -= f(row, "margin_bone_fraction") * params["margin_weight"]
    score += f(row, "continuity_score") * params["continuity_bonus"]
    return score


def select_case(candidates: list[dict[str, str]], params: dict[str, float]) -> tuple[dict[str, str], str]:
    model = [row for row in candidates if row.get("candidate_source") != "teacher_baseline"]
    teacher = next((row for row in candidates if row.get("candidate_source") == "teacher_baseline"), None)
    scored = sorted(model, key=lambda row: decision_score(row, params), reverse=True)
    selected = scored[0] if scored else teacher
    reason = "choose_highest_tuned_score"

    if selected is not None and (
        f(selected, "bone_overlap") > params["hard_bone_overlap"]
        or f(selected, "near_bone_fraction") > params["hard_near"]
    ):
        safer = [
            row
            for row in scored
            if f(row, "bone_overlap") <= params["safe_bone_overlap"]
            and f(row, "near_bone_fraction") <= params["safe_near"]
            and decision_score(row, params) >= decision_score(selected, params) - params["safe_score_gap"]
        ]
        if safer:
            selected = safer[0]
            reason = "choose_safer_topk_candidate"
        elif teacher is not None and params["allow_teacher_fallback"]:
            selected = teacher
            reason = "fallback_teacher_high_bone_risk"

    if (
        teacher is not None
        and selected is not None
        and selected.get("candidate_source") == "current_multibone"
        and decision_score(selected, params) < params["margin_shift_score"]
        and f(selected, "teacher_distance_mm") > params["margin_shift_teacher_distance"]
        and f(selected, "margin_bone_fraction") > params["margin_shift_margin"]
    ):
        selected = teacher
        reason = "fallback_teacher_possible_bone_margin_shift"

    return selected, reason


def evaluate(selection: list[dict[str, object]]) -> dict[str, object]:
    errors = [float(row["center_error_mm"]) for row in selection]
    ious = [float(row.get("bbox_iou") or row.get("pred_box_doctor_bbox_iou") or 0.0) for row in selection]
    coverages = [float(row["doctor_roi_coverage"]) for row in selection]
    bones = [float(row.get("bone_overlap") or row.get("pred_bone_overlap") or 0.0) for row in selection]
    return {
        "mean_center_error_mm": round(mean(errors), 3),
        "median_center_error_mm": round(median(errors), 3),
        "worst_center_error_mm": round(max(errors), 3),
        "mean_bbox_iou": round(mean(ious), 4),
        "mean_doctor_roi_coverage": round(mean(coverages), 4),
        "mean_pred_bone_overlap": round(mean(bones), 4),
        "objective": round(mean(errors) + 0.25 * max(errors) - 8.0 * mean(coverages) + 35.0 * mean(bones), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune final candidate selection from saved top-k candidates.")
    parser.add_argument("--candidate-csv", default="outputs/multibone_next_round/results/per_case_topk.csv")
    parser.add_argument("--teacher-csv", default="outputs/teacher_10cases/evaluation/ct_tendon_locator_results.csv")
    parser.add_argument("--output-dir", default="outputs/multibone_tuned_selection")
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidate_csv))
    teachers = {row["case"]: teacher_candidate(row) for row in read_csv(Path(args.teacher_csv))}
    cases = sorted({row["case"] for row in candidates})
    by_case = {case: [row for row in candidates if row["case"] == case] + ([teachers[case]] if case in teachers else []) for case in cases}

    search_space = {
        "current_bonus": [0.0, 0.2, 0.4],
        "teacher_refine_penalty": [0.6, 1.0, 1.4],
        "low_z_penalty": [0.0, 0.3],
        "teacher_bonus": [-0.2, 0.0, 0.2],
        "band_weight_delta": [-0.2, 0.0, 0.2],
        "near_weight": [0.0, 2.0],
        "margin_weight": [0.0, 1.0],
        "continuity_bonus": [0.0, 0.15],
        "hard_bone_overlap": [0.010, 0.012],
        "hard_near": [0.10, 0.12],
        "safe_bone_overlap": [0.006, 0.008],
        "safe_near": [0.08, 0.10],
        "safe_score_gap": [0.3, 0.5],
        "margin_shift_score": [3.5, 3.7],
        "margin_shift_teacher_distance": [6.0, 7.0],
        "margin_shift_margin": [0.035, 0.045],
        "allow_teacher_fallback": [1.0],
    }
    params = {
        "current_bonus": 0.2,
        "teacher_refine_penalty": 1.0,
        "low_z_penalty": 0.3,
        "teacher_bonus": 0.0,
        "band_weight_delta": 0.0,
        "near_weight": 2.0,
        "margin_weight": 1.0,
        "continuity_bonus": 0.0,
        "hard_bone_overlap": 0.012,
        "hard_near": 0.12,
        "safe_bone_overlap": 0.008,
        "safe_near": 0.10,
        "safe_score_gap": 0.5,
        "margin_shift_score": 3.7,
        "margin_shift_teacher_distance": 7.0,
        "margin_shift_margin": 0.035,
        "allow_teacher_fallback": 1.0,
    }
    output_dir = Path(args.output_dir)
    all_rows: list[dict[str, object]] = []
    best_selection: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_params: dict[str, float] | None = None

    def run_params(run_id: int, trial_params: dict[str, float]) -> dict[str, object]:
        selection: list[dict[str, object]] = []
        for case in cases:
            chosen, reason = select_case(by_case[case], trial_params)
            out = dict(chosen)
            out["decision_reason"] = reason
            out["tuned_decision_score"] = round(decision_score(chosen, trial_params), 4) if chosen.get("candidate_source") != "teacher_baseline" else trial_params["teacher_bonus"]
            selection.append(out)
        summary = evaluate(selection)
        row = {"run_id": run_id, **trial_params, **summary}
        for case in FOCUS_CASES:
            chosen = next(item for item in selection if item["case"] == case)
            row[f"{case}_source"] = chosen["candidate_source"]
            row[f"{case}_error"] = chosen["center_error_mm"]
            row[f"{case}_coverage"] = chosen["doctor_roi_coverage"]
            row[f"{case}_bone"] = chosen.get("bone_overlap") or chosen.get("pred_bone_overlap")
        row["_selection"] = selection
        return row

    run_id = 0
    for _round in range(2):
        improved = True
        while improved:
            improved = False
            for key, values in search_space.items():
                local_best = None
                for value in values:
                    trial = dict(params)
                    trial[key] = value
                    run_id += 1
                    row = run_params(run_id, trial)
                    all_rows.append({k: v for k, v in row.items() if k != "_selection"})
                    if local_best is None or float(row["objective"]) < float(local_best["objective"]):
                        local_best = row
                if local_best is not None and float(local_best["objective"]) < float(run_params(-1, params)["objective"]):
                    params = {key_name: local_best[key_name] for key_name in search_space}
                    improved = True
                    selection = local_best["_selection"]
                    summary = {k: local_best[k] for k in ("mean_center_error_mm", "median_center_error_mm", "worst_center_error_mm", "mean_bbox_iou", "mean_doctor_roi_coverage", "mean_pred_bone_overlap", "objective")}
                    if best_summary is None or float(summary["objective"]) < float(best_summary["objective"]):
                        best_summary = summary
                        best_selection = selection
                        best_params = dict(params)

    if best_summary is None:
        final_row = run_params(run_id + 1, params)
        best_summary = {k: final_row[k] for k in ("mean_center_error_mm", "median_center_error_mm", "worst_center_error_mm", "mean_bbox_iou", "mean_doctor_roi_coverage", "mean_pred_bone_overlap", "objective")}
        best_selection = final_row["_selection"]
        best_params = dict(params)

    for candidate in sorted(all_rows, key=lambda row: float(row["objective"]))[:5]:
        if best_summary is None or float(candidate["objective"]) < float(best_summary["objective"]):
            trial_params = {key: candidate[key] for key in search_space}
            final_row = run_params(run_id + 1, trial_params)
            summary = {k: final_row[k] for k in ("mean_center_error_mm", "median_center_error_mm", "worst_center_error_mm", "mean_bbox_iou", "mean_doctor_roi_coverage", "mean_pred_bone_overlap", "objective")}
            if float(summary["objective"]) < float(best_summary["objective"]):
                best_summary = summary
                best_selection = final_row["_selection"]
                best_params = trial_params

    ranked = sorted(all_rows, key=lambda row: float(row["objective"]))
    write_csv(output_dir / "selection_tuning_summary.csv", ranked)
    write_csv(output_dir / "selection_tuning_best_by_objective.csv", ranked[:20])
    write_csv(output_dir / "selection_tuning_best_final_selection.csv", best_selection)
    write_csv(output_dir / "selection_tuning_best_params.csv", [{**best_params, **best_summary}])

    source_results = Path(args.candidate_csv).parents[1]
    package_dir = output_dir / "best_package_seed"
    package_dir.mkdir(parents=True, exist_ok=True)
    for name in ("previews", "reports"):
        src = source_results / name
        dst = package_dir / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    print(f"best objective: {best_summary}")
    print(f"wrote tuning outputs to {output_dir}")


if __name__ == "__main__":
    main()
