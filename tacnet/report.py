"""Generate every numerical table used by the TAC-Net manuscript.

This module reads frozen experiment evidence. It never trains a model and it
never reads raw datasets. The separation is intentional: reviewers obtain the
exact reported values, while ``experiment.py`` remains the retraining path.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
OUTPUT = ROOT / "outputs" / "paper_tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main_results() -> list[dict[str, object]]:
    source = read_csv(EVIDENCE / "expected" / "phase3_dataset_summary.csv")
    paired = read_csv(EVIDENCE / "expected" / "phase3_paired_statistics.csv")
    names = {"credit": "Credit", "diabetes": "Diabetes", "nf_toniot_sanitized": "NF-ToN-IoT", "nf_unsw": "NF-UNSW", "unsw": "UNSW-NB15"}
    by_dataset = {row["dataset"]: row for row in source}
    preserved = {}
    for key in names:
        values = [row for row in paired if row["dataset"] == key]
        passed = sum(row["accuracy_preserved_within_0.005"].lower() == "true" for row in values)
        preserved[key] = f"{passed}/{len(values)}"
    rows = []
    for key, label in names.items():
        row = by_dataset[key]
        metric = "BA" if row["headline_metric"] == "balanced_accuracy" else "F1"
        full = float(row["full_headline_mean"])
        tac = float(row["tac_headline_mean"])
        digits = 6 if key == "nf_unsw" else 4
        rows.append({"dataset": label, "full": f"{full:.4f} {metric}", "tac_net": f"{tac:.4f} {metric}", "difference": f"{tac-full:+.{digits}f}", "macs_saved_pct": f"{float(row['macs_saved_pct_mean']):.1f}", "mean_layer": f"{float(row['mean_layer_mean']):.2f}/6", "seeds_preserved": preserved[key]})
    return rows


def depth_results() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(EVIDENCE / "expected" / "lightweight_gate_depth_pilot.csv"):
        rows.append({"depth": row["depth"], "full_ba": f"{float(row['full_balanced_accuracy']):.4f}", "tac_ba": f"{float(row['tac_balanced_accuracy']):.4f}", "difference": f"{float(row['difference']):+.4f}", "ci95": f"[{float(row['bootstrap_ci95_low']):.4f}, {float(row['bootstrap_ci95_high']):.4f}]", "macs_saved_pct": f"{float(row['macs_saved_pct']):.1f}", "speedup": f"{float(row['measured_speedup']):.2f}"})
    return rows


def ablation_results() -> list[dict[str, object]]:
    source = read_csv(EVIDENCE / "phase2" / "phase2_three_seed_summary.csv")
    names = {"credit": "Credit", "diabetes": "Diabetes", "unsw": "UNSW-NB15"}
    return [{"dataset": names[row["dataset"]], "score_difference": f"{float(row['delta_vs_full_mean']):+.4f}", "macs_saved_pct": f"{float(row['macs_saved_pct_mean']):.1f}", "mean_layer": f"{float(row['mean_layer_mean']):.2f}/6", "premature_harm_pct": f"{100*float(row['premature_harm_share_mean']):.3f}"} for row in source if row["variant"] == "next_flip_hgb_stable"]


def baseline_results() -> list[dict[str, object]]:
    source = read_csv(EVIDENCE / "expected" / "method_results.csv")
    wanted = {"LogisticRegression": "LR", "HistGradientBoosting": "HGB", "RandomForest": "RF", "Plain feedforward ANN": "Plain ANN", "Multi-exit ANN, full depth": "Full exit ANN", "TAC-Net generalized value gate": "TAC-Net"}
    datasets = {"credit": "Credit", "diabetes": "Diabetes", "unsw": "UNSW-NB15"}
    rows = []
    for row in source:
        if row["dataset"] not in datasets or row["method"] not in wanted or row["depth"] not in ("", "6"):
            continue
        rows.append({"dataset": datasets[row["dataset"]], "model": wanted[row["method"]], "accuracy": f"{float(row['accuracy']):.4f}", "macro_f1": f"{float(row['f1_macro']):.4f}", "balanced_accuracy_or_recall": f"{float(row['balanced_accuracy']):.4f}", "nll": f"{float(row['nll']):.4f}"})
    return rows


def _comparison_row(dataset: str, depth: str, name: str, result: dict, macs_saved="", mean_layer="", speedup_full="", speedup_plain="") -> dict[str, object]:
    metrics = result["metrics"]
    latency = result.get("inference", {}).get("microseconds_per_row_median", "")
    return {
        "dataset": dataset,
        "depth": depth,
        "model": name,
        "accuracy": f"{metrics['accuracy']:.6f}",
        "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
        "macro_f1": f"{metrics['f1_macro']:.6f}",
        "macro_precision": f"{metrics['precision_macro']:.6f}",
        "macro_recall": f"{metrics['recall_macro']:.6f}",
        "mcc": f"{metrics['mcc']:.6f}",
        "nll": f"{metrics['nll']:.6f}",
        "latency_us_per_row": f"{latency:.4f}" if latency != "" else "",
        "macs_saved_pct": macs_saved,
        "mean_exit_layer": mean_layer,
        "speedup_vs_full_depth": speedup_full,
        "speedup_vs_plain_ann": speedup_plain,
    }


def write_benchmark_report(result_path: Path) -> Path:
    """Flatten a completed reviewer run into one readable comparison table."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = []
    for dataset, dataset_result in payload["datasets"].items():
        for name, result in dataset_result["classical_machine_learning"].items():
            rows.append(_comparison_row(dataset, "", name, result))
        for depth, experiment in dataset_result["depth_experiments"].items():
            rows.append(_comparison_row(dataset, depth, "Plain feedforward ANN", experiment["plain_feedforward_ann"]))
            rows.append(_comparison_row(dataset, depth, "Multi-exit ANN, full depth", experiment["proposed_full_depth_control"]))
            dynamic = experiment["proposed_dynamic_exit"]
            rows.append(_comparison_row(
                dataset,
                depth,
                "TAC-Net dynamic exit",
                dynamic,
                f"{dynamic['macs_saved_vs_adaptive_full_pct']:.3f}",
                f"{dynamic['mean_layer']:.3f}",
                f"{dynamic['measured_speedup_vs_full_depth']:.3f}" if "measured_speedup_vs_full_depth" in dynamic else "",
                f"{dynamic['measured_speedup_vs_plain_ann']:.3f}",
            ))
    output_path = result_path.with_name(result_path.stem + "_comparison.csv")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _paper_runs() -> dict[tuple[str, int], dict]:
    wanted = {"credit", "diabetes", "unsw", "nf_unsw", "nf_toniot_sanitized"}
    runs = {}
    for path in (EVIDENCE / "phase3").glob("phase3_*.json"):
        if "quick" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in payload.get("runs", []):
            key = (run["dataset"], int(run["seed"]))
            if key[0] in wanted:
                runs[key] = run
    return runs


def write_reproduction_report(result_path: Path) -> Path:
    """Compare fresh training outputs with the matching paper runs."""
    fresh_runs = json.loads(result_path.read_text(encoding="utf-8"))["runs"]
    paper_runs = _paper_runs()
    rows = []
    for run in fresh_runs:
        key = (run["dataset"], int(run["seed"]))
        if key not in paper_runs:
            raise KeyError(f"No paper run exists for dataset={key[0]}, seed={key[1]}")
        paper = paper_runs[key]
        metric = "f1_macro" if len(run["classes"]) > 2 else "balanced_accuracy"
        fresh_full = float(run["full_depth"]["metrics"][metric])
        fresh_tac = float(run["tac_net"]["metrics"][metric])
        paper_full = float(paper["full_depth"]["metrics"][metric])
        paper_tac = float(paper["tac_net"]["metrics"][metric])
        fresh_macs = float(run["tac_net"]["macs_saved_pct"])
        paper_macs = float(paper["tac_net"]["macs_saved_pct"])
        rows.append({
            "dataset": key[0],
            "seed": key[1],
            "headline_metric": metric,
            "paper_full_depth": f"{paper_full:.8f}",
            "rerun_full_depth": f"{fresh_full:.8f}",
            "full_absolute_difference": f"{abs(fresh_full-paper_full):.8f}",
            "paper_tac_net": f"{paper_tac:.8f}",
            "rerun_tac_net": f"{fresh_tac:.8f}",
            "tac_absolute_difference": f"{abs(fresh_tac-paper_tac):.8f}",
            "paper_macs_saved_pct": f"{paper_macs:.4f}",
            "rerun_macs_saved_pct": f"{fresh_macs:.4f}",
            "matches_paper_rounding": (
                round(fresh_full, 4) == round(paper_full, 4)
                and round(fresh_tac, 4) == round(paper_tac, 4)
                and round(fresh_macs, 1) == round(paper_macs, 1)
            ),
        })
    output_path = result_path.with_name(result_path.stem + "_reproduction_audit.csv")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    tables = {"table_main.csv": main_results(), "table_depth.csv": depth_results(), "table_ablation.csv": ablation_results(), "table_baselines.csv": baseline_results()}
    for filename, rows in tables.items():
        if not rows:
            raise ValueError(f"No rows produced for {filename}")
        write_csv(filename, rows)
        print(f"{filename}: {len(rows)} rows")


if __name__ == "__main__":
    main()
