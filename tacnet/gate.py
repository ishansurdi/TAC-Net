"""Phase-2 TAC-Net study: learn when the next ANN layer helps or harms.

Labels are used only to train/tune the gate on the calibration partition.
During test inference the gate sees probabilities, uncertainty, stability, depth,
and cost features only. Test labels are opened afterwards for evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent

from datasets import DATASETS
from model import (
    GeneralizedAdaptiveANN,
    Settings,
    class_credibility_thresholds,
    collect_adaptive_logits,
    compute_metrics,
    conformal_quantile,
    cumulative_macs,
    fit_decision_weights,
    fit_temperature,
    gate_features_numpy,
    headline_score,
    minimum_class_recall,
    normalize_weighted_probabilities,
    policy_cost,
    prepare_dataset,
    set_seed,
    softmax,
    train_adaptive,
)


OUTPUT_DIR = HERE.parent / "outputs" / "phase2"


class ConstantEventModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(len(features), self.probability)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class GateVariant:
    name: str
    target: str
    learner: str
    require_stability: bool = False


VARIANTS = (
    GateVariant("final_logistic", "final", "logistic"),
    GateVariant("next_flip_hgb", "next", "histogram_boosting"),
    GateVariant("next_flip_hgb_stable", "next", "histogram_boosting", True),
    GateVariant("any_future_hgb", "any_future", "histogram_boosting"),
    GateVariant("correctness_delta_hgb", "correctness_delta", "histogram_boosting"),
)


def fit_binary_model(features: np.ndarray, target: np.ndarray, learner: str, seed: int):
    rate = float(target.mean())
    if len(np.unique(target)) == 1:
        model = ConstantEventModel(rate)
    elif learner == "logistic":
        model = LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=600, solver="lbfgs",
            random_state=seed,
        ).fit(features, target)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.06,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=0.5,
            class_weight="balanced",
            random_state=seed,
        ).fit(features, target)
    probability = model.predict_proba(features)[:, 1]
    return model, {
        "positive_rows": int(target.sum()),
        "positive_rate": rate,
        "fit_roc_auc": (
            float(roc_auc_score(target, probability)) if len(np.unique(target)) > 1 else None
        ),
        "fit_average_precision": float(average_precision_score(target, probability)),
    }


def make_gate_rows(
    probabilities: list[np.ndarray],
    labels: np.ndarray,
    parts: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    depth = len(probabilities)
    predictions = np.stack([
        (values * parts["decision_weights"]).argmax(axis=1)
        for values in probabilities
    ], axis=1)
    correct = predictions == labels[:, None]
    features = []
    layer_ids = []
    previous = None
    targets = {key: [] for key in ("benefit_final", "harm_final", "benefit_next", "harm_next", "benefit_any", "harm_any", "current_correct", "next_correct")}
    for layer_index in range(depth - 1):
        features.append(gate_features_numpy(
            probabilities[layer_index], previous, layer_index, depth,
            parts["cumulative_costs"], parts["conformal_quantiles"][layer_index],
            parts["class_thresholds"][layer_index], parts["decision_weights"],
        ))
        layer_ids.append(np.full(len(labels), layer_index, dtype=np.int16))
        current = correct[:, layer_index]
        following = correct[:, layer_index + 1]
        final = correct[:, -1]
        any_future = correct[:, layer_index + 1:].any(axis=1)
        targets["benefit_final"].append((~current & final).astype(np.int8))
        targets["harm_final"].append((current & ~final).astype(np.int8))
        targets["benefit_next"].append((~current & following).astype(np.int8))
        targets["harm_next"].append((current & ~following).astype(np.int8))
        targets["benefit_any"].append((~current & any_future).astype(np.int8))
        targets["harm_any"].append((current & ~any_future).astype(np.int8))
        targets["current_correct"].append(current.astype(np.int8))
        targets["next_correct"].append(following.astype(np.int8))
        previous = probabilities[layer_index]
    return (
        np.concatenate(features),
        {key: np.concatenate(values) for key, values in targets.items()},
        np.concatenate(layer_ids),
    )


def fit_variant(variant: GateVariant, features: np.ndarray, targets: dict, seed: int) -> dict:
    scaler = StandardScaler().fit(features) if variant.learner == "logistic" else None
    fitted_features = scaler.transform(features) if scaler is not None else features
    if variant.target == "correctness_delta":
        current_model, current_summary = fit_binary_model(
            fitted_features, targets["current_correct"], variant.learner, seed
        )
        future_model, future_summary = fit_binary_model(
            fitted_features, targets["next_correct"], variant.learner, seed + 1
        )
        return {
            "variant": variant,
            "scaler": scaler,
            "positive_model": future_model,
            "negative_model": current_model,
            "fit_summary": {"next_correct": future_summary, "current_correct": current_summary},
        }
    prefix = {"final": "final", "next": "next", "any_future": "any"}[variant.target]
    positive_model, positive_summary = fit_binary_model(
        fitted_features, targets[f"benefit_{prefix}"], variant.learner, seed
    )
    negative_model, negative_summary = fit_binary_model(
        fitted_features, targets[f"harm_{prefix}"], variant.learner, seed + 1
    )
    return {
        "variant": variant,
        "scaler": scaler,
        "positive_model": positive_model,
        "negative_model": negative_model,
        "fit_summary": {"benefit": positive_summary, "harm": negative_summary},
    }


def gain(model: dict, features: np.ndarray) -> np.ndarray:
    transformed = model["scaler"].transform(features) if model["scaler"] is not None else features
    positive = model["positive_model"].predict_proba(transformed)[:, 1]
    negative = model["negative_model"].predict_proba(transformed)[:, 1]
    return positive - negative


def route(
    probabilities: list[np.ndarray],
    parts: dict,
    model: dict,
    cost_lambda: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = len(probabilities)
    rows = len(probabilities[0])
    answer = normalize_weighted_probabilities(probabilities[-1], parts["decision_weights"])
    exits = np.full(rows, depth, dtype=np.int16)
    active = np.ones(rows, dtype=bool)
    previous = None
    previous_prediction = None
    for layer_index in range(depth - 1):
        current = probabilities[layer_index]
        weighted = normalize_weighted_probabilities(current, parts["decision_weights"])
        current_prediction = weighted.argmax(axis=1)
        features = gate_features_numpy(
            current, previous, layer_index, depth, parts["cumulative_costs"],
            parts["conformal_quantiles"][layer_index], parts["class_thresholds"][layer_index],
            parts["decision_weights"],
        )
        value = gain(model, features)
        remaining = 1.0 - parts["cumulative_costs"][layer_index] / parts["cumulative_costs"][-1]
        set_size = (current >= (1.0 - parts["conformal_quantiles"][layer_index])).sum(axis=1)
        confidence = current.max(axis=1)
        safe = (set_size == 1) & (
            confidence >= parts["class_thresholds"][layer_index][current_prediction]
        )
        if model["variant"].require_stability:
            stable = (
                np.zeros(rows, dtype=bool) if previous_prediction is None
                else current_prediction == previous_prediction
            )
            safe &= stable
        take = active & safe & (value - cost_lambda * remaining <= threshold)
        answer[take] = weighted[take]
        exits[take] = layer_index + 1
        active &= ~take
        previous = current
        previous_prediction = current_prediction
    return answer, exits


def route_precomputed(
    probabilities: list[np.ndarray],
    parts: dict,
    gains: list[np.ndarray],
    safe_masks: list[np.ndarray],
    cost_lambda: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast tuning route: expensive model probabilities are computed once."""
    depth = len(probabilities)
    rows = len(probabilities[0])
    answer = normalize_weighted_probabilities(probabilities[-1], parts["decision_weights"])
    exits = np.full(rows, depth, dtype=np.int16)
    active = np.ones(rows, dtype=bool)
    for layer_index, current in enumerate(probabilities[:-1]):
        remaining = 1.0 - parts["cumulative_costs"][layer_index] / parts["cumulative_costs"][-1]
        take = active & safe_masks[layer_index] & (
            gains[layer_index] - cost_lambda * remaining <= threshold
        )
        if take.any():
            weighted = normalize_weighted_probabilities(current[take], parts["decision_weights"])
            answer[take] = weighted
            exits[take] = layer_index + 1
        active &= ~take
    return answer, exits


def tune_variant(probabilities: list[np.ndarray], labels: np.ndarray, parts: dict, model: dict, max_drop: float, max_recall_drop: float) -> dict:
    final = normalize_weighted_probabilities(probabilities[-1], parts["decision_weights"])
    full_prediction = final.argmax(axis=1)
    full_score = headline_score(labels, full_prediction, final.shape[1])
    full_min_recall = minimum_class_recall(labels, full_prediction)
    all_gain = []
    safe_masks = []
    previous = None
    previous_prediction = None
    for layer_index, current in enumerate(probabilities[:-1]):
        features = gate_features_numpy(
            current, previous, layer_index, len(probabilities), parts["cumulative_costs"],
            parts["conformal_quantiles"][layer_index], parts["class_thresholds"][layer_index],
            parts["decision_weights"],
        )
        all_gain.append(gain(model, features))
        prediction = (current * parts["decision_weights"]).argmax(axis=1)
        set_size = (current >= (1.0 - parts["conformal_quantiles"][layer_index])).sum(axis=1)
        confidence = current.max(axis=1)
        safe = (set_size == 1) & (
            confidence >= parts["class_thresholds"][layer_index][prediction]
        )
        if model["variant"].require_stability:
            safe &= (
                np.zeros(len(current), dtype=bool) if previous_prediction is None
                else prediction == previous_prediction
            )
        safe_masks.append(safe)
        previous = current
        previous_prediction = prediction
    candidates = []
    for cost_lambda in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        adjusted = np.concatenate([
            values - cost_lambda * (1.0 - parts["cumulative_costs"][index] / parts["cumulative_costs"][-1])
            for index, values in enumerate(all_gain)
        ])
        for threshold in np.unique(np.quantile(adjusted, np.linspace(0.0, 1.0, 41))):
            answer, exits = route_precomputed(
                probabilities, parts, all_gain, safe_masks, cost_lambda, float(threshold)
            )
            prediction = answer.argmax(axis=1)
            score = headline_score(labels, prediction, answer.shape[1])
            min_recall = minimum_class_recall(labels, prediction)
            cost_fraction, mean_layer = policy_cost(exits, parts["cumulative_costs"])
            candidates.append({
                "cost_lambda": cost_lambda, "threshold": float(threshold),
                "score": score, "minimum_class_recall": min_recall,
                "cost_fraction": cost_fraction, "mean_layer": mean_layer,
                "feasible": score >= full_score - max_drop and min_recall >= full_min_recall - max_recall_drop,
            })
    feasible = [item for item in candidates if item["feasible"]]
    selected = min(feasible or candidates, key=lambda item: (item["cost_fraction"], -item["score"], -item["minimum_class_recall"]))
    return {
        "full_score": full_score,
        "full_minimum_class_recall": full_min_recall,
        "feasible_candidates": len(feasible),
        "evaluated_candidates": len(candidates),
        "selected": selected,
    }


def flip_diagnostics(labels: np.ndarray, probabilities: list[np.ndarray], exits: np.ndarray, weights: np.ndarray) -> dict:
    predictions = np.stack([(values * weights).argmax(axis=1) for values in probabilities], axis=1)
    row = np.arange(len(labels))
    chosen = exits - 1
    chosen_correct = predictions[row, chosen] == labels
    final_correct = predictions[:, -1] == labels
    has_next = chosen < predictions.shape[1] - 1
    next_index = np.minimum(chosen + 1, predictions.shape[1] - 1)
    next_correct = predictions[row, next_index] == labels
    exited = exits < predictions.shape[1]
    prevented_spoil = exited & chosen_correct & has_next & ~next_correct
    premature_harm = exited & ~chosen_correct & final_correct
    return {
        "early_exit_rows": int(exited.sum()),
        "early_exit_share": float(exited.mean()),
        "chosen_layer_accuracy": float(chosen_correct.mean()),
        "prevented_immediate_correct_to_wrong_rows": int(prevented_spoil.sum()),
        "prevented_immediate_correct_to_wrong_share": float(prevented_spoil.mean()),
        "premature_wrong_when_final_correct_rows": int(premature_harm.sum()),
        "premature_wrong_when_final_correct_share": float(premature_harm.mean()),
    }


def oracle_upper_bound(labels: np.ndarray, probabilities: list[np.ndarray], parts: dict) -> dict:
    weighted = [normalize_weighted_probabilities(values, parts["decision_weights"]) for values in probabilities]
    predictions = np.stack([values.argmax(axis=1) for values in weighted], axis=1)
    correct = predictions == labels[:, None]
    has_correct = correct.any(axis=1)
    earliest = np.argmax(correct, axis=1) + 1
    exits = np.where(has_correct, earliest, len(probabilities)).astype(np.int16)
    rows = np.arange(len(labels))
    answer = np.stack(weighted)[exits - 1, rows]
    cost_fraction, mean_layer = policy_cost(exits, parts["cumulative_costs"])
    return {
        "metrics": compute_metrics(labels, answer.argmax(axis=1), answer),
        "cost_fraction": cost_fraction,
        "mean_layer": mean_layer,
        "note": "Label-aware test oracle; upper bound only, never a deployable result",
    }


def run_once(dataset_name: str, seed: int, args) -> dict:
    set_seed(seed)
    data = prepare_dataset(DATASETS[dataset_name], seed, args.max_train_rows, args.max_test_rows)
    settings = Settings(
        depth=args.depth, width=args.width, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, distillation_weight=args.distillation_weight,
    )
    model = GeneralizedAdaptiveANN(data["x_train"].shape[1], len(data["classes"]), settings)
    print(f"\n{dataset_name} seed={seed}: training shared depth-{args.depth} ANN", flush=True)
    model, training = train_adaptive(
        model, data["x_train"], data["y_train"], data["x_calibration"],
        data["y_calibration"], settings, seed,
    )
    calibration_logits = collect_adaptive_logits(model, data["x_calibration"], args.batch_size * 4)
    test_logits = collect_adaptive_logits(model, data["x_test"], args.batch_size * 4)
    fit_idx, tune_idx = train_test_split(
        np.arange(len(data["y_calibration"])), test_size=0.50, random_state=seed,
        stratify=data["y_calibration"],
    )
    temperatures = [fit_temperature(values[fit_idx], data["y_calibration"][fit_idx]) for values in calibration_logits]
    calibration_probabilities = [softmax(values, temp) for values, temp in zip(calibration_logits, temperatures)]
    test_probabilities = [softmax(values, temp) for values, temp in zip(test_logits, temperatures)]
    weights = fit_decision_weights(calibration_probabilities[-1][fit_idx], data["y_calibration"][fit_idx])
    costs = cumulative_macs(data["x_train"].shape[1], len(data["classes"]), settings, adaptive=True)
    quantiles, thresholds = [], []
    for probabilities in calibration_probabilities[:-1]:
        quantile = conformal_quantile(probabilities[fit_idx], data["y_calibration"][fit_idx], args.alpha)
        singleton = (probabilities[fit_idx] >= 1.0 - quantile).sum(axis=1) == 1
        quantiles.append(quantile)
        thresholds.append(class_credibility_thresholds(
            probabilities[fit_idx], data["y_calibration"][fit_idx], singleton,
            weights, args.target_precision,
        ))
    parts = {
        "decision_weights": weights,
        "cumulative_costs": costs,
        "conformal_quantiles": quantiles,
        "class_thresholds": thresholds,
    }
    x_gate, targets, _ = make_gate_rows(
        [values[fit_idx] for values in calibration_probabilities],
        data["y_calibration"][fit_idx], parts,
    )
    final_test = normalize_weighted_probabilities(test_probabilities[-1], weights)
    output = {
        "dataset": dataset_name,
        "seed": seed,
        "rows": {"train": len(data["y_train"]), "calibration": len(data["y_calibration"]), "test": len(data["y_test"])},
        "classes": data["classes"],
        "settings": vars(settings),
        "training": training,
        "full_depth": {"metrics": compute_metrics(data["y_test"], final_test.argmax(axis=1), final_test), "mean_layer": args.depth, "cost_fraction": 1.0},
        "oracle_upper_bound": oracle_upper_bound(data["y_test"], test_probabilities, parts),
        "variants": {},
    }
    for variant in VARIANTS:
        print(f"  fitting/tuning gate {variant.name}", flush=True)
        started = time.perf_counter()
        fitted = fit_variant(variant, x_gate, targets, seed)
        tuning = tune_variant(
            [values[tune_idx] for values in calibration_probabilities],
            data["y_calibration"][tune_idx], parts, fitted,
            args.max_score_drop, args.max_class_recall_drop,
        )
        selected = tuning["selected"]
        answer, exits = route(
            test_probabilities, parts, fitted, selected["cost_lambda"], selected["threshold"]
        )
        cost_fraction, mean_layer = policy_cost(exits, costs)
        output["variants"][variant.name] = {
            "definition": vars(variant),
            "fit_summary": fitted["fit_summary"],
            "tuning": tuning,
            "metrics": compute_metrics(data["y_test"], answer.argmax(axis=1), answer),
            "mean_layer": mean_layer,
            "median_layer": float(np.median(exits)),
            "p95_layer": float(np.quantile(exits, 0.95)),
            "cost_fraction": cost_fraction,
            "macs_saved_pct": 100.0 * (1.0 - cost_fraction),
            "exit_counts": np.bincount(exits, minlength=args.depth + 1)[1:].tolist(),
            "flip_diagnostics": flip_diagnostics(data["y_test"], test_probabilities, exits, weights),
            "gate_fit_and_tune_seconds": time.perf_counter() - started,
        }
    return output


def summarize(all_runs: dict) -> dict:
    summary = {}
    for dataset, runs in all_runs.items():
        names = list(runs[0]["variants"])
        dataset_summary = {}
        for name in names:
            records = [run["variants"][name] for run in runs]
            metric_name = "f1_macro" if len(runs[0]["classes"]) > 2 else "balanced_accuracy"
            dataset_summary[name] = {
                "headline_metric": metric_name,
                "headline_mean": float(np.mean([record["metrics"][metric_name] for record in records])),
                "headline_std": float(np.std([record["metrics"][metric_name] for record in records], ddof=1)) if len(records) > 1 else 0.0,
                "accuracy_mean": float(np.mean([record["metrics"]["accuracy"] for record in records])),
                "macro_f1_mean": float(np.mean([record["metrics"]["f1_macro"] for record in records])),
                "minimum_class_recall_mean": float(np.mean([record["metrics"]["minimum_class_recall"] for record in records])),
                "nll_mean": float(np.mean([record["metrics"]["nll"] for record in records])),
                "macs_saved_pct_mean": float(np.mean([record["macs_saved_pct"] for record in records])),
                "mean_layer_mean": float(np.mean([record["mean_layer"] for record in records])),
                "prevented_spoil_share_mean": float(np.mean([record["flip_diagnostics"]["prevented_immediate_correct_to_wrong_share"] for record in records])),
                "premature_harm_share_mean": float(np.mean([record["flip_diagnostics"]["premature_wrong_when_final_correct_share"] for record in records])),
            }
        summary[dataset] = dataset_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["credit", "diabetes", "unsw"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 73, 101])
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--distillation-weight", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--max-score-drop", type=float, default=0.005)
    parser.add_argument("--max-class-recall-drop", type=float, default=0.02)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "TAC-Net Phase 2 flip-aware gate comparison",
        "test_labels_used_for_gate_training_or_selection": False,
        "datasets": {},
    }
    output_path = OUTPUT_DIR / f"phase2_depth{args.depth}_seeds_{'-'.join(map(str, args.seeds))}.json"
    for dataset in args.datasets:
        output["datasets"][dataset] = []
        for seed in args.seeds:
            output["datasets"][dataset].append(run_once(dataset, seed, args))
            output["summary"] = summarize(output["datasets"])
            output_path.write_text(json.dumps(output, indent=2, default=float), encoding="utf-8")
            print(f"checkpointed {output_path}", flush=True)
    print(f"completed {output_path}")


if __name__ == "__main__":
    main()
