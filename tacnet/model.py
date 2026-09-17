"""
Generalized adaptive-depth ANN for tabular classification.

The network may have any depth D >= 2. Every hidden layer has an auxiliary
classifier. A single shared value-of-computation gate is trained across all
non-final layers and asks, for every row at layer l:

    need_next(x, l, D) = predicted_gain(x, l)
                         - lambda_cost * remaining_MAC_fraction(l, D)

Continue when need_next is above a calibrated threshold. Exit only when two
safety conditions also hold: the conformal prediction set is a singleton and
the predicted class clears a class-specific credibility threshold. The gate
uses normalized depth and cost, so the same procedure works for 6, 7, 8, 12,
or another number of hidden layers without prescribing fixed exit positions.

The experiment keeps the test partition untouched and reports three groups:
classical machine learning, a plain feedforward ANN, and the proposed adaptive
ANN. It also evaluates common early-exit rules from prior work on the same
multi-exit backbone.

Examples:
    python generalized_adaptive_ann.py --datasets credit --depths 6 8 12
    python generalized_adaptive_ann.py --datasets unsw --depths 6 --epochs 18
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent

from datasets import DATASETS, DatasetConfig, TabularEncoder, load_partitions, split_xy


RESULTS_DIR = HERE.parent / "outputs" / "generalized"
GATE_FEATURE_NAMES = (
    "confidence",
    "margin",
    "normalized_entropy",
    "normalized_conformal_set_size",
    "credibility_gap",
    "prediction_changed",
    "probability_drift",
    "normalized_depth",
    "cumulative_cost_fraction",
    "remaining_cost_fraction",
)


@dataclass(frozen=True)
class Settings:
    depth: int
    width: int = 128
    dropout: float = 0.20
    epochs: int = 20
    patience: int = 4
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    focal_gamma: float = 1.5
    distillation_weight: float = 0.10


@dataclass
class GatePolicy:
    temperatures: list[float]
    decision_weights: np.ndarray
    conformal_quantiles: list[float]
    class_thresholds: list[np.ndarray]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    benefit_coefficients: np.ndarray
    benefit_intercept: float
    harm_coefficients: np.ndarray
    harm_intercept: float
    cost_lambda: float
    decision_threshold: float
    cumulative_costs: list[int]
    calibration_summary: dict

    def public_dict(self) -> dict:
        return {
            "formula": (
                "continue iff safety guard fails OR "
                "predicted_gain(z_l) - cost_lambda*(1-C_l/C_D) > decision_threshold"
            ),
            "features": list(GATE_FEATURE_NAMES),
            "temperatures": self.temperatures,
            "decision_weights": self.decision_weights.tolist(),
            "conformal_quantiles": self.conformal_quantiles,
            "class_thresholds": [values.tolist() for values in self.class_thresholds],
            "standardization_mean": self.scaler_mean.tolist(),
            "standardization_scale": self.scaler_scale.tolist(),
            "benefit_coefficients": self.benefit_coefficients.tolist(),
            "benefit_intercept": self.benefit_intercept,
            "harm_coefficients": self.harm_coefficients.tolist(),
            "harm_intercept": self.harm_intercept,
            "cost_lambda": self.cost_lambda,
            "decision_threshold": self.decision_threshold,
            "cumulative_costs": self.cumulative_costs,
            "calibration_summary": self.calibration_summary,
        }


class HiddenBlock(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class PlainFeedForwardANN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, settings: Settings):
        super().__init__()
        blocks = []
        current_dim = input_dim
        for _ in range(settings.depth):
            blocks.append(HiddenBlock(current_dim, settings.width, settings.dropout))
            current_dim = settings.width
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(settings.width, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        for block in self.blocks:
            hidden = block(hidden)
        return self.classifier(hidden)


class GeneralizedAdaptiveANN(nn.Module):
    """One depth-D ANN with a refinement classifier after every hidden layer."""

    def __init__(self, input_dim: int, num_classes: int, settings: Settings):
        super().__init__()
        blocks = []
        heads = []
        current_dim = input_dim
        for layer_index in range(settings.depth):
            blocks.append(HiddenBlock(current_dim, settings.width, settings.dropout))
            head_input = settings.width if layer_index == 0 else settings.width + num_classes
            heads.append(nn.Linear(head_input, num_classes))
            current_dim = settings.width
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        hidden = features
        previous_logits = None
        outputs = []
        for layer_index, (block, head) in enumerate(zip(self.blocks, self.heads)):
            hidden = block(hidden)
            if layer_index == 0:
                logits = head(hidden)
            else:
                previous_probabilities = previous_logits.softmax(dim=1)
                logits = head(torch.cat([hidden, previous_probabilities], dim=1))
            outputs.append(logits)
            previous_logits = logits
        return outputs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    log_probabilities = torch.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    rows = torch.arange(len(labels), device=labels.device)
    true_probability = probabilities[rows, labels]
    return (
        -weights[labels]
        * (1.0 - true_probability).pow(gamma)
        * log_probabilities[rows, labels]
    ).mean()


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    probabilities = softmax(logits, temperature)
    true_probability = np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1.0)
    return float(-np.log(true_probability).mean())


def train_plain(
    model: PlainFeedForwardANN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    settings: Settings,
    seed: int,
) -> tuple[PlainFeedForwardANN, dict]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    weights = class_weights(y_train, model.classifier.out_features)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=0,
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    started = time.perf_counter()

    for epoch in range(settings.epochs):
        model.train()
        running_loss = 0.0
        rows_seen = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = focal_loss(model(batch_x), batch_y, weights, settings.focal_gamma)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch_y)
            rows_seen += len(batch_y)

        validation_logits = collect_plain_logits(model, x_validation, settings.batch_size * 4)
        validation_nll = nll(validation_logits, y_validation)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(rows_seen, 1),
                "validation_nll": validation_nll,
            }
        )
        print(
            f"    plain epoch={epoch + 1:02d} validation_nll={validation_nll:.5f}",
            flush=True,
        )
        if validation_nll < best_loss - 1e-4:
            best_loss = validation_nll
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "training_seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
        "best_validation_nll": best_loss,
        "history": history,
    }


def train_adaptive(
    model: GeneralizedAdaptiveANN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    settings: Settings,
    seed: int,
) -> tuple[GeneralizedAdaptiveANN, dict]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    weights = class_weights(y_train, model.num_classes)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=0,
    )
    exit_weights = np.linspace(0.5, 1.0, settings.depth)
    exit_weights /= exit_weights.sum()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    started = time.perf_counter()

    for epoch in range(settings.epochs):
        model.train()
        running_loss = 0.0
        rows_seen = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            supervised = sum(
                float(weight)
                * focal_loss(exit_logits, batch_y, weights, settings.focal_gamma)
                for weight, exit_logits in zip(exit_weights, logits)
            )
            teacher = logits[-1].detach().softmax(dim=1)
            distillation = sum(
                torch.nn.functional.kl_div(
                    torch.log_softmax(exit_logits, dim=1), teacher, reduction="batchmean"
                )
                for exit_logits in logits[:-1]
            ) / max(len(logits) - 1, 1)
            loss = supervised + settings.distillation_weight * distillation
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch_y)
            rows_seen += len(batch_y)

        validation_logits = collect_adaptive_logits(
            model, x_validation, settings.batch_size * 4
        )
        validation_nll = float(
            np.mean([nll(exit_logits, y_validation) for exit_logits in validation_logits])
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(rows_seen, 1),
                "validation_nll": validation_nll,
            }
        )
        print(
            f"    adaptive epoch={epoch + 1:02d} validation_nll={validation_nll:.5f}",
            flush=True,
        )
        if validation_nll < best_loss - 1e-4:
            best_loss = validation_nll
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "training_seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
        "best_validation_nll": best_loss,
        "history": history,
    }


@torch.inference_mode()
def collect_plain_logits(
    model: PlainFeedForwardANN, features: np.ndarray, batch_size: int
) -> np.ndarray:
    model.eval()
    parts = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False
    )
    for (batch_x,) in loader:
        parts.append(model(batch_x).cpu().numpy())
    return np.concatenate(parts)


@torch.inference_mode()
def collect_adaptive_logits(
    model: GeneralizedAdaptiveANN, features: np.ndarray, batch_size: int
) -> list[np.ndarray]:
    model.eval()
    parts = [[] for _ in model.blocks]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False
    )
    for (batch_x,) in loader:
        for layer_index, logits in enumerate(model(batch_x)):
            parts[layer_index].append(logits.cpu().numpy())
    return [np.concatenate(layer_parts) for layer_parts in parts]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.linspace(0.25, 3.0, 24)
    losses = [nll(logits, labels, float(candidate)) for candidate in candidates]
    return float(candidates[int(np.argmin(losses))])


def normalize_weighted_probabilities(
    probabilities: np.ndarray, decision_weights: np.ndarray
) -> np.ndarray:
    weighted = probabilities * decision_weights
    return weighted / np.maximum(weighted.sum(axis=1, keepdims=True), 1e-12)


def fit_decision_weights(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    num_classes = probabilities.shape[1]
    weights = np.ones(num_classes, dtype=np.float64)
    best_score = headline_score(labels, probabilities.argmax(axis=1), num_classes)
    candidates = np.exp(np.linspace(np.log(0.3), np.log(3.0), 21))
    for _ in range(2):
        improved = False
        for class_index in range(num_classes):
            selected = weights[class_index]
            for candidate in candidates:
                weights[class_index] = candidate
                predictions = (probabilities * weights).argmax(axis=1)
                score = headline_score(labels, predictions, num_classes)
                if score > best_score + 1e-9:
                    best_score = score
                    selected = float(candidate)
                    improved = True
            weights[class_index] = selected
        if not improved:
            break
    return weights


def conformal_quantile(probabilities: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    scores = 1.0 - probabilities[np.arange(len(labels)), labels]
    level = np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores)
    return float(np.quantile(scores, min(level, 1.0), method="higher"))


def class_credibility_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    singleton: np.ndarray,
    decision_weights: np.ndarray,
    target_precision: float,
) -> np.ndarray:
    predicted = (probabilities * decision_weights).argmax(axis=1)
    confidence = probabilities.max(axis=1)
    thresholds = np.full(probabilities.shape[1], 1.01, dtype=np.float64)
    for class_index in range(probabilities.shape[1]):
        mask = singleton & (predicted == class_index)
        if mask.sum() < 20:
            continue
        class_confidence = confidence[mask]
        class_correct = (predicted[mask] == labels[mask]).astype(np.float64)
        order = np.argsort(-class_confidence)
        running_precision = np.cumsum(class_correct[order]) / np.arange(1, mask.sum() + 1)
        acceptable = np.flatnonzero(running_precision >= target_precision)
        if len(acceptable):
            thresholds[class_index] = float(class_confidence[order][acceptable[-1]])
    return thresholds


def cumulative_macs(
    input_dim: int, num_classes: int, settings: Settings, adaptive: bool
) -> list[int]:
    costs = []
    cumulative = 0
    current_dim = input_dim
    for layer_index in range(settings.depth):
        cumulative += current_dim * settings.width
        if adaptive:
            head_input = settings.width if layer_index == 0 else settings.width + num_classes
            cumulative += head_input * num_classes
            costs.append(cumulative)
        current_dim = settings.width
    if not adaptive:
        cumulative += settings.width * num_classes
        costs = [cumulative]
    return costs


def gate_features_numpy(
    probabilities: np.ndarray,
    previous_probabilities: np.ndarray | None,
    layer_index: int,
    depth: int,
    cumulative_costs: list[int],
    conformal_quantile_value: float,
    class_thresholds: np.ndarray,
    decision_weights: np.ndarray,
) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -np.sum(
        np.clip(probabilities, 1e-12, 1.0)
        * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    ) / max(math.log(probabilities.shape[1]), 1e-12)
    set_size = (probabilities >= (1.0 - conformal_quantile_value)).sum(axis=1)
    predicted = (probabilities * decision_weights).argmax(axis=1)
    credibility_gap = confidence - class_thresholds[predicted]
    if previous_probabilities is None:
        prediction_changed = np.zeros(len(probabilities))
        probability_drift = np.zeros(len(probabilities))
    else:
        prediction_changed = (
            (probabilities * decision_weights).argmax(axis=1)
            != (previous_probabilities * decision_weights).argmax(axis=1)
        ).astype(np.float64)
        probability_drift = np.abs(probabilities - previous_probabilities).sum(axis=1)
    cost_fraction = cumulative_costs[layer_index] / cumulative_costs[-1]
    return np.column_stack(
        [
            confidence,
            margin,
            entropy,
            set_size / probabilities.shape[1],
            credibility_gap,
            prediction_changed,
            probability_drift,
            np.full(len(probabilities), (layer_index + 1) / depth),
            np.full(len(probabilities), cost_fraction),
            np.full(len(probabilities), 1.0 - cost_fraction),
        ]
    ).astype(np.float64)


def fit_gain_model(
    probabilities: list[np.ndarray],
    labels: np.ndarray,
    policy_parts: dict,
    depth: int,
) -> tuple[StandardScaler, np.ndarray, float, np.ndarray, float, dict]:
    final_predictions = (
        probabilities[-1] * policy_parts["decision_weights"]
    ).argmax(axis=1)
    final_correct = (final_predictions == labels).astype(np.float64)
    features = []
    benefit_targets = []
    harm_targets = []
    previous = None
    for layer_index in range(depth - 1):
        current = probabilities[layer_index]
        current_predictions = (
            current * policy_parts["decision_weights"]
        ).argmax(axis=1)
        current_correct = (current_predictions == labels).astype(np.float64)
        features.append(
            gate_features_numpy(
                current,
                previous,
                layer_index,
                depth,
                policy_parts["cumulative_costs"],
                policy_parts["conformal_quantiles"][layer_index],
                policy_parts["class_thresholds"][layer_index],
                policy_parts["decision_weights"],
            )
        )
        benefit_targets.append(((current_correct == 0) & (final_correct == 1)).astype(int))
        harm_targets.append(((current_correct == 1) & (final_correct == 0)).astype(int))
        previous = current
    x_gate = np.concatenate(features)
    y_benefit = np.concatenate(benefit_targets)
    y_harm = np.concatenate(harm_targets)
    scaler = StandardScaler().fit(x_gate)
    standardized = scaler.transform(x_gate)

    def fit_event(target: np.ndarray) -> tuple[np.ndarray, float, dict]:
        rate = float(target.mean())
        if len(np.unique(target)) == 1:
            clipped = float(np.clip(rate, 1e-6, 1.0 - 1e-6))
            coefficients = np.zeros(standardized.shape[1], dtype=np.float64)
            intercept = float(np.log(clipped / (1.0 - clipped)))
            return coefficients, intercept, {
                "positive_rows": int(target.sum()),
                "positive_rate": rate,
                "roc_auc_on_gate_fit": None,
                "average_precision_on_gate_fit": rate,
            }
        model = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
        ).fit(standardized, target)
        probability = model.predict_proba(standardized)[:, 1]
        return (
            np.asarray(model.coef_[0], dtype=np.float64),
            float(model.intercept_[0]),
            {
                "positive_rows": int(target.sum()),
                "positive_rate": rate,
                "roc_auc_on_gate_fit": float(roc_auc_score(target, probability)),
                "average_precision_on_gate_fit": float(
                    average_precision_score(target, probability)
                ),
            },
        )

    benefit_coefficients, benefit_intercept, benefit_summary = fit_event(y_benefit)
    harm_coefficients, harm_intercept, harm_summary = fit_event(y_harm)
    summary = {
        "rows": int(len(x_gate)),
        "benefit_event": benefit_summary,
        "harm_event": harm_summary,
    }
    return (
        scaler,
        benefit_coefficients,
        benefit_intercept,
        harm_coefficients,
        harm_intercept,
        summary,
    )


def expected_gain_numpy(features: np.ndarray, policy: GatePolicy) -> np.ndarray:
    standardized = (features - policy.scaler_mean) / policy.scaler_scale
    benefit_logit = standardized @ policy.benefit_coefficients + policy.benefit_intercept
    harm_logit = standardized @ policy.harm_coefficients + policy.harm_intercept
    benefit_probability = 1.0 / (1.0 + np.exp(-np.clip(benefit_logit, -30.0, 30.0)))
    harm_probability = 1.0 / (1.0 + np.exp(-np.clip(harm_logit, -30.0, 30.0)))
    return benefit_probability - harm_probability


def route_probabilities(
    probabilities: list[np.ndarray], policy: GatePolicy
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(probabilities[0])
    depth = len(probabilities)
    final = normalize_weighted_probabilities(probabilities[-1], policy.decision_weights)
    answers = final.copy()
    exit_layers = np.full(rows, depth, dtype=np.int16)
    active = np.ones(rows, dtype=bool)
    previous = None
    for layer_index in range(depth - 1):
        current = probabilities[layer_index]
        features = gate_features_numpy(
            current,
            previous,
            layer_index,
            depth,
            policy.cumulative_costs,
            policy.conformal_quantiles[layer_index],
            policy.class_thresholds[layer_index],
            policy.decision_weights,
        )
        predicted_gain = expected_gain_numpy(features, policy)
        remaining_cost = 1.0 - policy.cumulative_costs[layer_index] / policy.cumulative_costs[-1]
        need_next = predicted_gain - policy.cost_lambda * remaining_cost
        set_size = (current >= (1.0 - policy.conformal_quantiles[layer_index])).sum(axis=1)
        confidence = current.max(axis=1)
        predicted = (current * policy.decision_weights).argmax(axis=1)
        safe = (
            (set_size == 1)
            & (confidence >= policy.class_thresholds[layer_index][predicted])
        )
        take = active & safe & (need_next <= policy.decision_threshold)
        answers[take] = normalize_weighted_probabilities(
            current[take], policy.decision_weights
        )
        exit_layers[take] = layer_index + 1
        active &= ~take
        previous = current
    return answers, exit_layers


def headline_score(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    if num_classes > 2:
        return float(f1_score(labels, predictions, average="macro", zero_division=0))
    return float(balanced_accuracy_score(labels, predictions))


def minimum_class_recall(labels: np.ndarray, predictions: np.ndarray) -> float:
    values = recall_score(labels, predictions, average=None, zero_division=0)
    return float(values.min())


def compute_metrics(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict:
    matrix = confusion_matrix(labels, predictions)
    recalls = recall_score(labels, predictions, average=None, zero_division=0)
    precisions = precision_score(labels, predictions, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "nll": float(
            -np.log(
                np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1.0)
            ).mean()
        ),
        "minimum_class_recall": float(recalls.min()),
        "per_class_precision": precisions.tolist(),
        "per_class_recall": recalls.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def policy_cost(exit_layers: np.ndarray, cumulative_costs: list[int]) -> tuple[float, float]:
    costs = np.array([cumulative_costs[layer - 1] for layer in exit_layers], dtype=np.float64)
    return float(costs.mean() / cumulative_costs[-1]), float(exit_layers.mean())


def tune_gate_policy(
    probabilities: list[np.ndarray],
    labels: np.ndarray,
    policy_parts: dict,
    scaler: StandardScaler,
    benefit_coefficients: np.ndarray,
    benefit_intercept: float,
    harm_coefficients: np.ndarray,
    harm_intercept: float,
    max_score_drop: float,
    max_class_recall_drop: float,
) -> tuple[float, float, dict]:
    depth = len(probabilities)
    final_probabilities = normalize_weighted_probabilities(
        probabilities[-1], policy_parts["decision_weights"]
    )
    final_predictions = final_probabilities.argmax(axis=1)
    full_score = headline_score(labels, final_predictions, final_probabilities.shape[1])
    full_min_recall = minimum_class_recall(labels, final_predictions)
    base_policy = GatePolicy(
        temperatures=policy_parts["temperatures"],
        decision_weights=policy_parts["decision_weights"],
        conformal_quantiles=policy_parts["conformal_quantiles"],
        class_thresholds=policy_parts["class_thresholds"],
        scaler_mean=scaler.mean_,
        scaler_scale=np.maximum(scaler.scale_, 1e-12),
        benefit_coefficients=benefit_coefficients,
        benefit_intercept=benefit_intercept,
        harm_coefficients=harm_coefficients,
        harm_intercept=harm_intercept,
        cost_lambda=0.0,
        decision_threshold=0.0,
        cumulative_costs=policy_parts["cumulative_costs"],
        calibration_summary={},
    )

    gains = []
    previous = None
    for layer_index in range(depth - 1):
        features = gate_features_numpy(
            probabilities[layer_index],
            previous,
            layer_index,
            depth,
            policy_parts["cumulative_costs"],
            policy_parts["conformal_quantiles"][layer_index],
            policy_parts["class_thresholds"][layer_index],
            policy_parts["decision_weights"],
        )
        gains.append(expected_gain_numpy(features, base_policy))
        previous = probabilities[layer_index]

    candidates = []
    for cost_lambda in (0.0, 0.01, 0.02, 0.05, 0.10):
        adjusted = np.concatenate(
            [
                values
                - cost_lambda
                * (
                    1.0
                    - policy_parts["cumulative_costs"][layer_index]
                    / policy_parts["cumulative_costs"][-1]
                )
                for layer_index, values in enumerate(gains)
            ]
        )
        thresholds = np.unique(
            np.quantile(adjusted, np.linspace(0.0, 1.0, 31)).astype(np.float64)
        )
        thresholds = np.concatenate(([-1e9], thresholds))
        for threshold in thresholds:
            trial = GatePolicy(
                temperatures=policy_parts["temperatures"],
                decision_weights=policy_parts["decision_weights"],
                conformal_quantiles=policy_parts["conformal_quantiles"],
                class_thresholds=policy_parts["class_thresholds"],
                scaler_mean=scaler.mean_,
                scaler_scale=np.maximum(scaler.scale_, 1e-12),
                benefit_coefficients=benefit_coefficients,
                benefit_intercept=benefit_intercept,
                harm_coefficients=harm_coefficients,
                harm_intercept=harm_intercept,
                cost_lambda=cost_lambda,
                decision_threshold=float(threshold),
                cumulative_costs=policy_parts["cumulative_costs"],
                calibration_summary={},
            )
            routed_probabilities, exit_layers = route_probabilities(probabilities, trial)
            predictions = routed_probabilities.argmax(axis=1)
            score = headline_score(labels, predictions, routed_probabilities.shape[1])
            min_recall = minimum_class_recall(labels, predictions)
            cost_fraction, mean_layer = policy_cost(
                exit_layers, policy_parts["cumulative_costs"]
            )
            feasible = (
                score >= full_score - max_score_drop
                and min_recall >= full_min_recall - max_class_recall_drop
            )
            candidates.append(
                {
                    "cost_lambda": cost_lambda,
                    "threshold": float(threshold),
                    "score": score,
                    "minimum_class_recall": min_recall,
                    "cost_fraction": cost_fraction,
                    "mean_layer": mean_layer,
                    "feasible": feasible,
                }
            )

    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    selected = min(
        feasible or candidates,
        key=lambda candidate: (
            candidate["cost_fraction"],
            -candidate["score"],
            -candidate["minimum_class_recall"],
        ),
    )
    summary = {
        "full_depth_score": full_score,
        "full_depth_minimum_class_recall": full_min_recall,
        "max_score_drop": max_score_drop,
        "max_class_recall_drop": max_class_recall_drop,
        "selected": selected,
        "feasible_candidates": len(feasible),
        "evaluated_candidates": len(candidates),
    }
    return selected["cost_lambda"], selected["threshold"], summary


def calibrate_gate(
    calibration_logits: list[np.ndarray],
    calibration_labels: np.ndarray,
    cumulative_costs: list[int],
    alpha: float,
    target_precision: float,
    max_score_drop: float,
    max_class_recall_drop: float,
    seed: int,
) -> GatePolicy:
    fit_indices, tune_indices = train_test_split(
        np.arange(len(calibration_labels)),
        test_size=0.50,
        random_state=seed,
        stratify=calibration_labels,
    )
    temperatures = [
        fit_temperature(logits[fit_indices], calibration_labels[fit_indices])
        for logits in calibration_logits
    ]
    fit_probabilities = [
        softmax(logits[fit_indices], temperature)
        for logits, temperature in zip(calibration_logits, temperatures)
    ]
    tune_probabilities = [
        softmax(logits[tune_indices], temperature)
        for logits, temperature in zip(calibration_logits, temperatures)
    ]
    decision_weights = fit_decision_weights(
        fit_probabilities[-1], calibration_labels[fit_indices]
    )
    conformal_quantiles = []
    class_thresholds = []
    for probabilities in fit_probabilities[:-1]:
        quantile = conformal_quantile(
            probabilities, calibration_labels[fit_indices], alpha
        )
        singleton = (probabilities >= (1.0 - quantile)).sum(axis=1) == 1
        conformal_quantiles.append(quantile)
        class_thresholds.append(
            class_credibility_thresholds(
                probabilities,
                calibration_labels[fit_indices],
                singleton,
                decision_weights,
                target_precision,
            )
        )

    policy_parts = {
        "temperatures": temperatures,
        "decision_weights": decision_weights,
        "conformal_quantiles": conformal_quantiles,
        "class_thresholds": class_thresholds,
        "cumulative_costs": cumulative_costs,
    }
    (
        scaler,
        benefit_coefficients,
        benefit_intercept,
        harm_coefficients,
        harm_intercept,
        gain_summary,
    ) = fit_gain_model(
        fit_probabilities,
        calibration_labels[fit_indices],
        policy_parts,
        len(calibration_logits),
    )
    cost_lambda, decision_threshold, tuning_summary = tune_gate_policy(
        tune_probabilities,
        calibration_labels[tune_indices],
        policy_parts,
        scaler,
        benefit_coefficients,
        benefit_intercept,
        harm_coefficients,
        harm_intercept,
        max_score_drop,
        max_class_recall_drop,
    )
    return GatePolicy(
        temperatures=temperatures,
        decision_weights=decision_weights,
        conformal_quantiles=conformal_quantiles,
        class_thresholds=class_thresholds,
        scaler_mean=scaler.mean_,
        scaler_scale=np.maximum(scaler.scale_, 1e-12),
        benefit_coefficients=benefit_coefficients,
        benefit_intercept=benefit_intercept,
        harm_coefficients=harm_coefficients,
        harm_intercept=harm_intercept,
        cost_lambda=cost_lambda,
        decision_threshold=decision_threshold,
        cumulative_costs=cumulative_costs,
        calibration_summary={
            "gate_fit_rows": int(len(fit_indices)),
            "gate_tune_rows": int(len(tune_indices)),
            "gain_model": gain_summary,
            "policy_tuning": tuning_summary,
        },
    )


def gate_features_torch(
    probabilities: torch.Tensor,
    previous_probabilities: torch.Tensor | None,
    layer_index: int,
    policy: GatePolicy,
    thresholds: torch.Tensor,
    decision_weights: torch.Tensor,
) -> torch.Tensor:
    ordered, _ = torch.sort(probabilities, dim=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -torch.sum(
        probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log(), dim=1
    ) / max(math.log(probabilities.shape[1]), 1e-12)
    set_size = (
        probabilities >= (1.0 - policy.conformal_quantiles[layer_index])
    ).sum(dim=1)
    predicted = (probabilities * decision_weights).argmax(dim=1)
    credibility_gap = confidence - thresholds[predicted]
    if previous_probabilities is None:
        changed = torch.zeros_like(confidence)
        drift = torch.zeros_like(confidence)
    else:
        changed = (
            (probabilities * decision_weights).argmax(dim=1)
            != (previous_probabilities * decision_weights).argmax(dim=1)
        ).float()
        drift = torch.abs(probabilities - previous_probabilities).sum(dim=1)
    cost_fraction = policy.cumulative_costs[layer_index] / policy.cumulative_costs[-1]
    return torch.stack(
        [
            confidence,
            margin,
            entropy,
            set_size.float() / probabilities.shape[1],
            credibility_gap,
            changed,
            drift,
            torch.full_like(confidence, (layer_index + 1) / len(policy.temperatures)),
            torch.full_like(confidence, cost_fraction),
            torch.full_like(confidence, 1.0 - cost_fraction),
        ],
        dim=1,
    )


@torch.inference_mode()
def dynamic_predict(
    model: GeneralizedAdaptiveANN,
    features: np.ndarray,
    policy: GatePolicy,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute only the layers required by each active row."""
    model.eval()
    output = np.zeros((len(features), model.num_classes), dtype=np.float32)
    exit_layers = np.zeros(len(features), dtype=np.int16)
    temperatures = [torch.tensor(value, dtype=torch.float32) for value in policy.temperatures]
    decision_weights = torch.tensor(policy.decision_weights, dtype=torch.float32)
    thresholds = [torch.tensor(values, dtype=torch.float32) for values in policy.class_thresholds]
    scaler_mean = torch.tensor(policy.scaler_mean, dtype=torch.float32)
    scaler_scale = torch.tensor(policy.scaler_scale, dtype=torch.float32)
    benefit_coefficients = torch.tensor(
        policy.benefit_coefficients, dtype=torch.float32
    )
    benefit_intercept = torch.tensor(policy.benefit_intercept, dtype=torch.float32)
    harm_coefficients = torch.tensor(policy.harm_coefficients, dtype=torch.float32)
    harm_intercept = torch.tensor(policy.harm_intercept, dtype=torch.float32)

    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        active_rows = torch.arange(stop - start)
        hidden = torch.from_numpy(features[start:stop])
        previous_head_probabilities = None
        previous_gate_probabilities = None
        batch_output = torch.zeros((stop - start, model.num_classes), dtype=torch.float32)
        batch_exits = torch.zeros(stop - start, dtype=torch.int16)

        for layer_index, (block, head) in enumerate(zip(model.blocks, model.heads)):
            hidden = block(hidden)
            if layer_index == 0:
                logits = head(hidden)
            else:
                logits = head(torch.cat([hidden, previous_head_probabilities], dim=1))
            head_probabilities = torch.softmax(logits, dim=1)
            probabilities = torch.softmax(logits / temperatures[layer_index], dim=1)
            weighted = probabilities * decision_weights
            weighted = weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)

            if layer_index == len(model.blocks) - 1:
                batch_output[active_rows] = weighted
                batch_exits[active_rows] = layer_index + 1
                break

            features_for_gate = gate_features_torch(
                probabilities,
                previous_gate_probabilities,
                layer_index,
                policy,
                thresholds[layer_index],
                decision_weights,
            )
            standardized = (features_for_gate - scaler_mean) / scaler_scale
            benefit_probability = torch.sigmoid(
                standardized @ benefit_coefficients + benefit_intercept
            )
            harm_probability = torch.sigmoid(
                standardized @ harm_coefficients + harm_intercept
            )
            predicted_gain = benefit_probability - harm_probability
            remaining_cost = (
                1.0
                - policy.cumulative_costs[layer_index] / policy.cumulative_costs[-1]
            )
            need_next = predicted_gain - policy.cost_lambda * remaining_cost
            set_size = (
                probabilities >= (1.0 - policy.conformal_quantiles[layer_index])
            ).sum(dim=1)
            confidence = probabilities.max(dim=1).values
            predicted = (probabilities * decision_weights).argmax(dim=1)
            safe = (set_size == 1) & (confidence >= thresholds[layer_index][predicted])
            take = safe & (need_next <= policy.decision_threshold)

            if take.any():
                exiting_rows = active_rows[take]
                batch_output[exiting_rows] = weighted[take]
                batch_exits[exiting_rows] = layer_index + 1
            keep = ~take
            active_rows = active_rows[keep]
            hidden = hidden[keep]
            previous_head_probabilities = head_probabilities[keep]
            previous_gate_probabilities = probabilities[keep]

        output[start:stop] = batch_output.numpy()
        exit_layers[start:stop] = batch_exits.numpy()
    return output, exit_layers


def inference_profile(function, rows: int, repeats: int = 5) -> dict:
    function()
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - started)
    per_row = np.array(elapsed) / rows
    return {
        "microseconds_per_row_median": float(np.median(per_row) * 1e6),
        "microseconds_per_row_p95": float(np.quantile(per_row, 0.95) * 1e6),
        "throughput_rows_per_second": float(rows / np.median(elapsed)),
        "repeats": repeats,
    }


def paired_bootstrap(
    labels: np.ndarray,
    proposed: np.ndarray,
    comparison: np.ndarray,
    num_classes: int,
    seed: int,
    resamples: int = 300,
) -> dict:
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for index in range(resamples):
        rows = rng.integers(0, len(labels), len(labels))
        differences[index] = headline_score(
            labels[rows], proposed[rows], num_classes
        ) - headline_score(labels[rows], comparison[rows], num_classes)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "verdict": "better" if low > 0 else ("worse" if high < 0 else "tied"),
        "resamples": resamples,
    }


def tune_prior_policy(
    name: str,
    tune_probabilities: list[np.ndarray],
    tune_labels: np.ndarray,
    test_probabilities: list[np.ndarray],
    policy: GatePolicy,
    max_score_drop: float,
) -> dict:
    num_classes = tune_probabilities[0].shape[1]
    final_tune = normalize_weighted_probabilities(
        tune_probabilities[-1], policy.decision_weights
    )
    full_score = headline_score(tune_labels, final_tune.argmax(axis=1), num_classes)

    def score_values(probabilities):
        if name == "max_softmax":
            return probabilities.max(axis=1)
        if name == "margin":
            ordered = np.sort(probabilities, axis=1)
            return ordered[:, -1] - ordered[:, -2]
        clipped = np.clip(probabilities, 1e-12, 1.0)
        return np.sum(clipped * np.log(clipped), axis=1)

    pooled = np.concatenate([score_values(values) for values in tune_probabilities[:-1]])
    thresholds = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, 31)))

    def route(probabilities, threshold):
        rows = len(probabilities[0])
        answer = normalize_weighted_probabilities(
            probabilities[-1], policy.decision_weights
        )
        exits = np.full(rows, len(probabilities), dtype=np.int16)
        active = np.ones(rows, dtype=bool)
        for layer_index, current in enumerate(probabilities[:-1]):
            take = active & (score_values(current) >= threshold)
            answer[take] = normalize_weighted_probabilities(
                current[take], policy.decision_weights
            )
            exits[take] = layer_index + 1
            active &= ~take
        return answer, exits

    candidates = []
    for threshold in thresholds:
        answer, exits = route(tune_probabilities, threshold)
        score = headline_score(tune_labels, answer.argmax(axis=1), num_classes)
        cost, mean_layer = policy_cost(exits, policy.cumulative_costs)
        if score >= full_score - max_score_drop:
            candidates.append((cost, -score, threshold, mean_layer))
    if candidates:
        _, _, threshold, _ = min(candidates)
    else:
        threshold = 1e9
    answer, exits = route(test_probabilities, threshold)
    predictions = answer.argmax(axis=1)
    cost, mean_layer = policy_cost(exits, policy.cumulative_costs)
    return {
        "rule": name,
        "calibrated_threshold": float(threshold),
        "probabilities": answer,
        "predictions": predictions,
        "exit_layers": exits,
        "mean_layer": mean_layer,
        "cost_fraction": cost,
    }


def fixed_prior_policy(
    name: str,
    probabilities: list[np.ndarray],
    policy: GatePolicy,
) -> dict:
    rows = len(probabilities[0])
    answer = normalize_weighted_probabilities(
        probabilities[-1], policy.decision_weights
    )
    exits = np.full(rows, len(probabilities), dtype=np.int16)
    active = np.ones(rows, dtype=bool)
    previous_predictions = None
    for layer_index, current in enumerate(probabilities[:-1]):
        weighted = normalize_weighted_probabilities(current, policy.decision_weights)
        predictions = weighted.argmax(axis=1)
        if name == "conformal_singleton":
            take = active & (
                (current >= (1.0 - policy.conformal_quantiles[layer_index])).sum(axis=1)
                == 1
            )
        elif previous_predictions is None:
            take = np.zeros(rows, dtype=bool)
        else:
            take = active & (predictions == previous_predictions)
        answer[take] = weighted[take]
        exits[take] = layer_index + 1
        active &= ~take
        previous_predictions = predictions
    cost, mean_layer = policy_cost(exits, policy.cumulative_costs)
    return {
        "rule": name,
        "probabilities": answer,
        "predictions": answer.argmax(axis=1),
        "exit_layers": exits,
        "mean_layer": mean_layer,
        "cost_fraction": cost,
    }


def baseline_models(seed: int) -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=800, class_weight="balanced", solver="lbfgs"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=31,
            class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=160, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=160, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=seed
        ),
    }


def fit_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    latency_rows: int,
) -> dict:
    results = {}
    sample = x_test[: min(latency_rows, len(x_test))]
    for name, model in baseline_models(seed).items():
        print(f"  fitting baseline {name}", flush=True)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - started
        probabilities = np.asarray(model.predict_proba(x_test), dtype=np.float64)
        predictions = probabilities.argmax(axis=1)
        profile = inference_profile(lambda: model.predict_proba(sample), len(sample), repeats=3)
        results[name] = {
            "metrics": compute_metrics(y_test, predictions, probabilities),
            "training_seconds": training_seconds,
            "inference": profile,
        }
    return results


def stratified_limit(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    indices = np.arange(len(labels))
    if limit is None or limit >= len(indices):
        return indices
    selected, _ = train_test_split(
        indices, train_size=limit, random_state=seed, stratify=labels
    )
    return np.sort(selected)


def prepare_dataset(
    dataset: DatasetConfig,
    seed: int,
    max_train_rows: int | None,
    max_test_rows: int | None,
) -> dict:
    train_frame, calibration_frame, test_frame = load_partitions(dataset)
    x_train_frame, y_train_text = split_xy(train_frame, dataset)
    x_calibration_frame, y_calibration_text = split_xy(calibration_frame, dataset)
    x_test_frame, y_test_text = split_xy(test_frame, dataset)
    label_encoder = LabelEncoder().fit(y_train_text)
    y_train = label_encoder.transform(y_train_text)
    y_calibration = label_encoder.transform(y_calibration_text)
    y_test = label_encoder.transform(y_test_text)
    train_indices = stratified_limit(y_train, max_train_rows, seed)
    test_indices = stratified_limit(y_test, max_test_rows, seed)
    x_train_frame = x_train_frame.iloc[train_indices].reset_index(drop=True)
    y_train = y_train[train_indices]
    x_test_frame = x_test_frame.iloc[test_indices].reset_index(drop=True)
    y_test = y_test[test_indices]
    encoder = TabularEncoder(dataset.categorical, leaf_components=2, seed=seed).fit_raw(
        x_train_frame
    )
    return {
        "x_train": encoder.raw_matrix(x_train_frame).astype(np.float32),
        "y_train": y_train,
        "x_calibration": encoder.raw_matrix(x_calibration_frame).astype(np.float32),
        "y_calibration": y_calibration,
        "x_test": encoder.raw_matrix(x_test_frame).astype(np.float32),
        "y_test": y_test,
        "classes": label_encoder.classes_.tolist(),
        "input_columns": x_train_frame.columns.tolist(),
    }


def run_depth(
    dataset_name: str,
    data: dict,
    settings: Settings,
    arguments,
) -> dict:
    print(f"\n{dataset_name}: depth={settings.depth}", flush=True)
    x_train = data["x_train"]
    y_train = data["y_train"]
    fit_indices, validation_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=0.10,
        random_state=arguments.seed,
        stratify=y_train,
    )
    num_classes = len(data["classes"])

    plain_seed = arguments.seed + settings.depth * 10
    adaptive_seed = plain_seed + 1
    set_seed(plain_seed)
    plain = PlainFeedForwardANN(x_train.shape[1], num_classes, settings)
    plain, plain_training = train_plain(
        plain,
        x_train[fit_indices],
        y_train[fit_indices],
        x_train[validation_indices],
        y_train[validation_indices],
        settings,
        plain_seed,
    )
    set_seed(adaptive_seed)
    adaptive = GeneralizedAdaptiveANN(x_train.shape[1], num_classes, settings)
    adaptive, adaptive_training = train_adaptive(
        adaptive,
        x_train[fit_indices],
        y_train[fit_indices],
        x_train[validation_indices],
        y_train[validation_indices],
        settings,
        adaptive_seed,
    )

    calibration_logits = collect_adaptive_logits(
        adaptive, data["x_calibration"], settings.batch_size * 4
    )
    adaptive_costs = cumulative_macs(
        x_train.shape[1], num_classes, settings, adaptive=True
    )
    policy = calibrate_gate(
        calibration_logits,
        data["y_calibration"],
        adaptive_costs,
        arguments.alpha,
        arguments.target_precision,
        arguments.max_score_drop,
        arguments.max_class_recall_drop,
        arguments.seed,
    )

    plain_test_logits = collect_plain_logits(
        plain, data["x_test"], settings.batch_size * 4
    )
    plain_temperature = fit_temperature(
        collect_plain_logits(plain, data["x_calibration"], settings.batch_size * 4),
        data["y_calibration"],
    )
    plain_probabilities = softmax(plain_test_logits, plain_temperature)
    plain_predictions = plain_probabilities.argmax(axis=1)

    adaptive_test_logits = collect_adaptive_logits(
        adaptive, data["x_test"], settings.batch_size * 4
    )
    adaptive_test_probabilities = [
        softmax(logits, temperature)
        for logits, temperature in zip(adaptive_test_logits, policy.temperatures)
    ]
    full_probabilities = normalize_weighted_probabilities(
        adaptive_test_probabilities[-1], policy.decision_weights
    )
    full_predictions = full_probabilities.argmax(axis=1)
    dynamic_probabilities, exit_layers = dynamic_predict(
        adaptive, data["x_test"], policy, arguments.inference_batch_size
    )
    dynamic_predictions = dynamic_probabilities.argmax(axis=1)
    simulated_probabilities, simulated_exit_layers = route_probabilities(
        adaptive_test_probabilities, policy
    )
    simulated_predictions = simulated_probabilities.argmax(axis=1)
    routing_verification = {
        "prediction_match_rate": float(
            np.mean(dynamic_predictions == simulated_predictions)
        ),
        "exit_layer_match_rate": float(
            np.mean(exit_layers == simulated_exit_layers)
        ),
        "maximum_probability_difference": float(
            np.max(np.abs(dynamic_probabilities - simulated_probabilities))
        ),
    }
    if routing_verification["prediction_match_rate"] < 0.999:
        raise RuntimeError(
            "dynamic inference does not reproduce the calibrated routing simulation"
        )

    calibration_probabilities = [
        softmax(logits, temperature)
        for logits, temperature in zip(calibration_logits, policy.temperatures)
    ]
    _, tune_indices = train_test_split(
        np.arange(len(data["y_calibration"])),
        test_size=0.50,
        random_state=arguments.seed,
        stratify=data["y_calibration"],
    )
    prior_rules = {}
    for rule_name in ("max_softmax", "negative_entropy", "margin"):
        result = tune_prior_policy(
            rule_name,
            [values[tune_indices] for values in calibration_probabilities],
            data["y_calibration"][tune_indices],
            adaptive_test_probabilities,
            policy,
            arguments.max_score_drop,
        )
        result["metrics"] = compute_metrics(
            data["y_test"], result.pop("predictions"), result["probabilities"]
        )
        result.pop("probabilities")
        result.pop("exit_layers")
        prior_rules[rule_name] = result
    for rule_name in ("conformal_singleton", "cross_stage_agreement"):
        result = fixed_prior_policy(rule_name, adaptive_test_probabilities, policy)
        result["metrics"] = compute_metrics(
            data["y_test"], result.pop("predictions"), result["probabilities"]
        )
        result.pop("probabilities")
        result.pop("exit_layers")
        prior_rules[rule_name] = result

    profile_rows = min(arguments.latency_rows, len(data["x_test"]))
    profile_x = data["x_test"][:profile_rows]
    plain_profile = inference_profile(
        lambda: collect_plain_logits(plain, profile_x, arguments.inference_batch_size),
        profile_rows,
        repeats=arguments.latency_repeats,
    )
    adaptive_profile = inference_profile(
        lambda: dynamic_predict(
            adaptive, profile_x, policy, arguments.inference_batch_size
        ),
        profile_rows,
        repeats=arguments.latency_repeats,
    )
    full_depth_profile = inference_profile(
        lambda: collect_adaptive_logits(
            adaptive, profile_x, arguments.inference_batch_size
        )[-1],
        profile_rows,
        repeats=arguments.latency_repeats,
    )
    cost_fraction, mean_layer = policy_cost(exit_layers, adaptive_costs)
    exit_counts = np.bincount(exit_layers, minlength=settings.depth + 1)[1:]
    plain_macs = cumulative_macs(
        x_train.shape[1], num_classes, settings, adaptive=False
    )[-1]
    mean_adaptive_macs = float(
        np.mean([adaptive_costs[layer - 1] for layer in exit_layers])
    )

    return {
        "settings": asdict(settings),
        "formula": policy.public_dict(),
        "plain_feedforward_ann": {
            "metrics": compute_metrics(
                data["y_test"], plain_predictions, plain_probabilities
            ),
            "training": plain_training,
            "inference": plain_profile,
            "estimated_macs_per_row": int(plain_macs),
        },
        "proposed_full_depth_control": {
            "metrics": compute_metrics(
                data["y_test"], full_predictions, full_probabilities
            ),
            "training": adaptive_training,
            "inference": full_depth_profile,
            "mean_layer": float(settings.depth),
            "estimated_macs_per_row": int(adaptive_costs[-1]),
        },
        "proposed_dynamic_exit": {
            "metrics": compute_metrics(
                data["y_test"], dynamic_predictions, dynamic_probabilities
            ),
            "inference": adaptive_profile,
            "mean_layer": mean_layer,
            "median_layer": float(np.median(exit_layers)),
            "p95_layer": float(np.quantile(exit_layers, 0.95)),
            "exit_counts": exit_counts.tolist(),
            "exit_shares": (exit_counts / len(exit_layers)).tolist(),
            "adaptive_full_cost_fraction": cost_fraction,
            "macs_saved_vs_adaptive_full_pct": 100.0 * (1.0 - cost_fraction),
            "estimated_macs_per_row": mean_adaptive_macs,
            "estimated_cost_vs_plain_ann": mean_adaptive_macs / plain_macs,
            "measured_speedup_vs_plain_ann": (
                plain_profile["microseconds_per_row_median"]
                / adaptive_profile["microseconds_per_row_median"]
            ),
            "measured_speedup_vs_full_depth": (
                full_depth_profile["microseconds_per_row_median"]
                / adaptive_profile["microseconds_per_row_median"]
            ),
            "routing_verification": routing_verification,
        },
        "prior_exit_rules_same_backbone": prior_rules,
        "significance": {
            "dynamic_vs_plain_ann": paired_bootstrap(
                data["y_test"],
                dynamic_predictions,
                plain_predictions,
                num_classes,
                arguments.seed + 101,
                arguments.bootstrap_resamples,
            ),
            "dynamic_vs_same_network_full_depth": paired_bootstrap(
                data["y_test"],
                dynamic_predictions,
                full_predictions,
                num_classes,
                arguments.seed + 202,
                arguments.bootstrap_resamples,
            ),
        },
    }


def run_dataset(dataset_name: str, arguments) -> dict:
    data = prepare_dataset(
        DATASETS[dataset_name],
        arguments.seed,
        arguments.max_train_rows,
        arguments.max_test_rows,
    )
    print(
        f"\n{'=' * 78}\n{dataset_name}: train={len(data['y_train']):,} "
        f"calibration={len(data['y_calibration']):,} test={len(data['y_test']):,} "
        f"features={data['x_train'].shape[1]} classes={len(data['classes'])}\n{'=' * 78}",
        flush=True,
    )
    results = {
        "dataset": dataset_name,
        "classes": data["classes"],
        "input_columns": data["input_columns"],
        "rows": {
            "train": len(data["y_train"]),
            "calibration": len(data["y_calibration"]),
            "test": len(data["y_test"]),
        },
        "classical_machine_learning": {},
        "depth_experiments": {},
    }
    if not arguments.skip_ml:
        results["classical_machine_learning"] = fit_baselines(
            data["x_train"],
            data["y_train"],
            data["x_test"],
            data["y_test"],
            arguments.seed,
            arguments.latency_rows,
        )

    for depth in arguments.depths:
        settings = Settings(
            depth=depth,
            width=arguments.width,
            dropout=arguments.dropout,
            epochs=arguments.epochs,
            patience=arguments.patience,
            batch_size=arguments.batch_size,
            distillation_weight=arguments.distillation_weight,
        )
        results["depth_experiments"][str(depth)] = run_depth(
            dataset_name, data, settings, arguments
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=["credit"]
    )
    parser.add_argument("--depths", nargs="+", type=int, default=[6, 8, 12])
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--distillation-weight", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--max-score-drop", type=float, default=0.005)
    parser.add_argument("--max-class-recall-drop", type=float, default=0.02)
    parser.add_argument("--latency-rows", type=int, default=4096)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=300)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    if any(depth < 2 for depth in arguments.depths):
        raise ValueError("every requested depth must be at least 2")
    if not 0.0 <= arguments.max_score_drop <= 0.10:
        raise ValueError("--max-score-drop must be between 0 and 0.10")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"seed_{arguments.seed}_depths_{'-'.join(map(str, arguments.depths))}"
    output_path = RESULTS_DIR / f"generalized_{run_id}.json"
    output = {
        "experiment": "Generalized adaptive-depth ANN",
        "decision_rule": (
            "Continue at layer l when the row is uncertain or when the learned expected "
            "gain of deeper computation exceeds its remaining normalized cost."
        ),
        "protocol": {
            "test_used_for_selection": False,
            "calibration_partition_split": "50% gate fitting / 50% policy tuning",
            "depths": arguments.depths,
            "seed": arguments.seed,
            "max_score_drop": arguments.max_score_drop,
            "max_class_recall_drop": arguments.max_class_recall_drop,
            "target_precision": arguments.target_precision,
        },
        "datasets": {},
    }
    for dataset_name in arguments.datasets:
        output["datasets"][dataset_name] = run_dataset(dataset_name, arguments)
        output_path.write_text(json.dumps(output, indent=2, default=float), encoding="utf-8")
        print(f"checkpointed {output_path}", flush=True)
    from report import write_benchmark_report
    comparison_path = write_benchmark_report(output_path)
    print(f"comparison table: {comparison_path}", flush=True)
    print(f"\nCompleted: {output_path}", flush=True)


if __name__ == "__main__":
    main()
