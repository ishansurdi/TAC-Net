"""Phase-3 TAC-Net: rare-class training and measured dynamic CPU execution.

This is additive: it reads existing datasets and writes only to phase3_results.
Architecture selection uses train/calibration data. Test labels are opened only
after the architecture and gate are frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent

from datasets import DATASETS
from gate import (
    GateVariant,
    fit_variant,
    flip_diagnostics,
    make_gate_rows,
    route,
    tune_variant,
)
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
    normalize_weighted_probabilities,
    policy_cost,
    prepare_dataset,
    set_seed,
    softmax,
)


RESULTS_DIR = HERE.parent / "outputs" / "phase3"
DEFAULT_GATE = GateVariant("next_flip_hgb_stable", "next", "histogram_boosting", True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_softmax_focal(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_counts: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    adjusted = logits + log_counts
    log_probabilities = torch.log_softmax(adjusted, dim=1)
    rows = torch.arange(len(labels), device=labels.device)
    true_probability = log_probabilities.exp()[rows, labels]
    return (-(1.0 - true_probability).pow(gamma) * log_probabilities[rows, labels]).mean()


def train_multiexit(
    model: GeneralizedAdaptiveANN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    settings: Settings,
    seed: int,
    loss_mode: str,
) -> tuple[GeneralizedAdaptiveANN, dict]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    counts = np.bincount(y_train, minlength=model.num_classes).astype(np.float64)
    log_counts = torch.tensor(np.log(np.maximum(counts, 1.0)), dtype=torch.float32)
    inverse = np.power(np.maximum(counts, 1.0), -0.65)
    inverse /= inverse.mean()
    inverse = np.minimum(inverse, 8.0)
    class_weights = torch.tensor(inverse, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=settings.batch_size, shuffle=True, num_workers=0,
    )
    exit_weights = np.linspace(0.5, 1.0, settings.depth)
    exit_weights /= exit_weights.sum()
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(settings.epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            losses = []
            for logits in outputs:
                if loss_mode == "balanced_softmax":
                    losses.append(balanced_softmax_focal(logits, batch_y, log_counts, settings.focal_gamma))
                else:
                    losses.append(torch.nn.functional.cross_entropy(logits, batch_y, weight=class_weights))
            supervised = sum(float(weight) * loss for weight, loss in zip(exit_weights, losses))
            teacher = outputs[-1].detach().softmax(dim=1)
            distillation = sum(
                torch.nn.functional.kl_div(torch.log_softmax(logits, dim=1), teacher, reduction="batchmean")
                for logits in outputs[:-1]
            ) / max(len(outputs) - 1, 1)
            loss = supervised + settings.distillation_weight * distillation
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch_y)
            seen += len(batch_y)
        logits = collect_adaptive_logits(model, x_validation, settings.batch_size * 4)[-1]
        probabilities = softmax(logits)
        predictions = probabilities.argmax(axis=1)
        score = headline_score(y_validation, predictions, model.num_classes)
        min_recall = compute_metrics(y_validation, predictions, probabilities)["minimum_class_recall"]
        selection_score = score + 0.05 * min_recall
        history.append({"epoch": epoch + 1, "train_loss": running / seen, "validation_headline": score, "validation_minimum_class_recall": min_recall})
        print(f"    {loss_mode} epoch={epoch+1:02d} val_score={score:.5f} min_recall={min_recall:.5f}", flush=True)
        if selection_score > best_score + 1e-4:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    model.load_state_dict(best_state)
    return model, {"loss_mode": loss_mode, "training_seconds": time.perf_counter() - started, "history": history, "best_selection_score": best_score}


def calibrate_components(calibration_logits, labels, costs, args, seed):
    fit_idx, tune_idx = train_test_split(np.arange(len(labels)), test_size=0.50, random_state=seed, stratify=labels)
    temperatures = [fit_temperature(values[fit_idx], labels[fit_idx]) for values in calibration_logits]
    probabilities = [softmax(values, temperature) for values, temperature in zip(calibration_logits, temperatures)]
    weights = fit_decision_weights(probabilities[-1][fit_idx], labels[fit_idx])
    quantiles, thresholds = [], []
    for values in probabilities[:-1]:
        q = conformal_quantile(values[fit_idx], labels[fit_idx], args.alpha)
        singleton = (values[fit_idx] >= 1.0 - q).sum(axis=1) == 1
        quantiles.append(q)
        thresholds.append(class_credibility_thresholds(values[fit_idx], labels[fit_idx], singleton, weights, args.target_precision))
    parts = {"decision_weights": weights, "cumulative_costs": costs, "conformal_quantiles": quantiles, "class_thresholds": thresholds}
    x_gate, targets, _ = make_gate_rows([values[fit_idx] for values in probabilities], labels[fit_idx], parts)
    gate_variant = GateVariant(
        f"next_flip_{args.gate_learner}_stable",
        "next",
        args.gate_learner,
        True,
    )
    gate = fit_variant(gate_variant, x_gate, targets, seed)
    tuning = tune_variant([values[tune_idx] for values in probabilities], labels[tune_idx], parts, gate, args.max_score_drop, args.max_class_recall_drop)
    return probabilities, parts, gate, tuning, temperatures, fit_idx, tune_idx


@torch.inference_mode()
def dynamic_predict(
    model: GeneralizedAdaptiveANN,
    features: np.ndarray,
    temperatures: list[float],
    parts: dict,
    gate: dict,
    selected: dict,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = np.zeros((len(features), model.num_classes), dtype=np.float32)
    exits = np.zeros(len(features), dtype=np.int16)
    weights = parts["decision_weights"]
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        active_global = np.arange(start, stop)
        hidden = torch.from_numpy(features[start:stop])
        previous_head = None
        previous_gate = None
        previous_prediction = None
        for layer_index, (block, head) in enumerate(zip(model.blocks, model.heads)):
            hidden = block(hidden)
            logits = head(hidden) if layer_index == 0 else head(torch.cat([hidden, previous_head], dim=1))
            raw_probability = torch.softmax(logits, dim=1)
            probability = softmax(logits.cpu().numpy(), temperatures[layer_index])
            weighted = normalize_weighted_probabilities(probability, weights)
            prediction = weighted.argmax(axis=1)
            if layer_index == len(model.blocks) - 1:
                output[active_global] = weighted
                exits[active_global] = layer_index + 1
                break
            feature_matrix = gate_features_numpy(
                probability, previous_gate, layer_index, len(model.blocks),
                parts["cumulative_costs"], parts["conformal_quantiles"][layer_index],
                parts["class_thresholds"][layer_index], weights,
            )
            transformed = gate["scaler"].transform(feature_matrix) if gate["scaler"] is not None else feature_matrix
            benefit = gate["positive_model"].predict_proba(transformed)[:, 1]
            harm = gate["negative_model"].predict_proba(transformed)[:, 1]
            remaining = 1.0 - parts["cumulative_costs"][layer_index] / parts["cumulative_costs"][-1]
            value = benefit - harm - selected["cost_lambda"] * remaining
            set_size = (probability >= 1.0 - parts["conformal_quantiles"][layer_index]).sum(axis=1)
            confidence = probability.max(axis=1)
            safe = (set_size == 1) & (confidence >= parts["class_thresholds"][layer_index][prediction])
            stable = np.zeros(len(prediction), dtype=bool) if previous_prediction is None else prediction == previous_prediction
            take = safe & stable & (value <= selected["threshold"])
            if take.any():
                output[active_global[take]] = weighted[take]
                exits[active_global[take]] = layer_index + 1
            keep = ~take
            active_global = active_global[keep]
            if len(active_global) == 0:
                break
            hidden = hidden[torch.from_numpy(keep)]
            previous_head = raw_probability[torch.from_numpy(keep)]
            previous_gate = probability[keep]
            previous_prediction = prediction[keep]
    return output, exits


def profile(function, rows: int, repeats: int) -> dict:
    function()
    values = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - started) / 1000.0 / rows)
    return {
        "microseconds_per_row_median": float(np.median(values)),
        "microseconds_per_row_p95": float(np.quantile(values, 0.95)),
        "microseconds_per_row_samples": [float(value) for value in values],
        "repeats": repeats,
        "rows": rows,
    }


def run(dataset_name: str, seed: int, args) -> dict:
    data = prepare_dataset(DATASETS[dataset_name], seed, args.max_train_rows, args.max_test_rows)
    settings = Settings(depth=args.depth, width=args.width, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, distillation_weight=args.distillation_weight)
    loss_modes = ["inverse_frequency"] if len(data["classes"]) == 2 else args.multiclass_loss_modes
    candidates = []
    for index, loss_mode in enumerate(loss_modes):
        print(f"\n{dataset_name} seed={seed}: training candidate {loss_mode}", flush=True)
        set_seed(seed + index * 1000)
        candidate = GeneralizedAdaptiveANN(data["x_train"].shape[1], len(data["classes"]), settings)
        candidate, training = train_multiexit(candidate, data["x_train"], data["y_train"], data["x_calibration"], data["y_calibration"], settings, seed + index * 1000, loss_mode)
        logits = collect_adaptive_logits(candidate, data["x_calibration"], args.batch_size * 4)
        final_prob = softmax(logits[-1])
        metrics = compute_metrics(data["y_calibration"], final_prob.argmax(axis=1), final_prob)
        selection = headline_score(data["y_calibration"], final_prob.argmax(axis=1), len(data["classes"])) + args.rare_recall_weight * metrics["minimum_class_recall"]
        candidates.append({"model": candidate, "training": training, "calibration_metrics": metrics, "selection_score": selection, "logits": logits})
    chosen_index = int(np.argmax([item["selection_score"] for item in candidates]))
    chosen = candidates[chosen_index]
    model = chosen["model"]
    calibration_logits = chosen["logits"]
    costs = cumulative_macs(data["x_train"].shape[1], len(data["classes"]), settings, adaptive=True)
    calibration_prob, parts, gate, tuning, temperatures, _, _ = calibrate_components(calibration_logits, data["y_calibration"], costs, args, seed)
    test_logits = collect_adaptive_logits(model, data["x_test"], args.batch_size * 4)
    test_probabilities = [softmax(values, temperature) for values, temperature in zip(test_logits, temperatures)]
    selected = tuning["selected"]
    offline_probability, offline_exits = route(test_probabilities, parts, gate, selected["cost_lambda"], selected["threshold"])
    dynamic_probability, dynamic_exits = dynamic_predict(model, data["x_test"], temperatures, parts, gate, selected, args.dynamic_batch_size)
    equivalence = {"prediction_match": float((offline_probability.argmax(axis=1) == dynamic_probability.argmax(axis=1)).mean()), "exit_match": float((offline_exits == dynamic_exits).mean()), "max_probability_difference": float(np.abs(offline_probability - dynamic_probability).max())}
    if equivalence["prediction_match"] < 1.0 or equivalence["exit_match"] < 1.0:
        raise RuntimeError(f"Offline/dynamic mismatch: {equivalence}")
    sample_rows = min(args.latency_rows, len(data["x_test"]))
    sample = data["x_test"][:sample_rows]
    dynamic_latency = profile(lambda: dynamic_predict(model, sample, temperatures, parts, gate, selected, args.dynamic_batch_size), sample_rows, args.latency_repeats)
    full_latency = profile(lambda: collect_adaptive_logits(model, sample, args.dynamic_batch_size), sample_rows, args.latency_repeats)
    metrics = compute_metrics(data["y_test"], dynamic_probability.argmax(axis=1), dynamic_probability)
    full_probability = normalize_weighted_probabilities(test_probabilities[-1], parts["decision_weights"])
    full_metrics = compute_metrics(data["y_test"], full_probability.argmax(axis=1), full_probability)
    cost_fraction, mean_layer = policy_cost(dynamic_exits, costs)
    artifact_tag = f"_{args.run_tag}" if args.run_tag else ""
    predictions_path = RESULTS_DIR / f"predictions_{dataset_name}_seed{seed}{artifact_tag}.npz"
    np.savez_compressed(predictions_path, labels=data["y_test"], full_predictions=full_probability.argmax(axis=1), tac_predictions=dynamic_probability.argmax(axis=1), exit_layers=dynamic_exits)
    return {
        "dataset": dataset_name, "seed": seed, "classes": data["classes"], "settings": vars(settings),
        "loss_candidates": [{"loss_mode": item["training"]["loss_mode"], "selection_score": item["selection_score"], "calibration_metrics": item["calibration_metrics"], "training_seconds": item["training"]["training_seconds"]} for item in candidates],
        "selected_loss_mode": chosen["training"]["loss_mode"], "gate": {"variant": vars(gate["variant"]), "tuning": tuning},
        "full_depth": {"metrics": full_metrics, "latency": full_latency, "mean_layer": args.depth},
        "tac_net": {"metrics": metrics, "latency": dynamic_latency, "mean_layer": mean_layer, "median_layer": float(np.median(dynamic_exits)), "p95_layer": float(np.quantile(dynamic_exits, 0.95)), "macs_saved_pct": 100.0 * (1.0 - cost_fraction), "measured_speedup_vs_full": full_latency["microseconds_per_row_median"] / dynamic_latency["microseconds_per_row_median"], "exit_counts": np.bincount(dynamic_exits, minlength=args.depth + 1)[1:].tolist(), "flip_diagnostics": flip_diagnostics(data["y_test"], test_probabilities, dynamic_exits, parts["decision_weights"]), "equivalence": equivalence},
        "prediction_artifact": {"path": str(predictions_path), "sha256": sha256(predictions_path)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["unsw"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dynamic-batch-size", type=int, default=512)
    parser.add_argument("--gate-learner", choices=["logistic", "histogram_boosting"], default=DEFAULT_GATE.learner)
    parser.add_argument("--run-tag", default="", help="Suffix for additive pilot outputs; prevents overwriting frozen results.")
    parser.add_argument("--distillation-weight", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--max-score-drop", type=float, default=0.005)
    parser.add_argument("--max-class-recall-drop", type=float, default=0.02)
    parser.add_argument("--rare-recall-weight", type=float, default=0.10)
    parser.add_argument("--multiclass-loss-modes", nargs="+", choices=["inverse_frequency", "balanced_softmax"], default=["inverse_frequency", "balanced_softmax"])
    parser.add_argument("--latency-rows", type=int, default=4096)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_tag = f"_{args.run_tag}" if args.run_tag else ""
    output_path = RESULTS_DIR / f"phase3_{'-'.join(args.datasets)}_seeds_{'-'.join(map(str,args.seeds))}{output_tag}.json"
    output = {"experiment": "TAC-Net Phase 3 runtime and rare-class study", "protocol": {"test_used_for_selection": False, "arguments": vars(args)}, "runs": []}
    for dataset in args.datasets:
        for seed in args.seeds:
            output["runs"].append(run(dataset, seed, args))
            output_path.write_text(json.dumps(output, indent=2, default=float), encoding="utf-8")
            print(f"checkpointed {output_path}", flush=True)
    print(f"completed {output_path}")


if __name__ == "__main__":
    main()
