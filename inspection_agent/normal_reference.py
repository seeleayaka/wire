"""Pure decision helpers for a multi-normal-reference image triage gate."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class NormalReferenceError(ValueError):
    """Raised when a normal-reference decision input is invalid."""


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise NormalReferenceError("descriptor must have a finite non-zero norm")
    return value / norm


def descriptor_from_patch_features(features: np.ndarray, *, grid_size: int = 2) -> np.ndarray:
    """Pool normalized DINO patch features globally and over a spatial grid."""
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 3 or min(value.shape) <= 0:
        raise NormalReferenceError("patch features must have shape [height, width, channels]")
    if grid_size < 1 or grid_size > min(value.shape[:2]):
        raise NormalReferenceError("grid_size must fit inside the patch grid")
    pooled = [value.mean(axis=(0, 1))]
    row_groups = np.array_split(np.arange(value.shape[0]), grid_size)
    column_groups = np.array_split(np.arange(value.shape[1]), grid_size)
    for rows in row_groups:
        for columns in column_groups:
            pooled.append(value[np.ix_(rows, columns)].mean(axis=(0, 1)))
    return _unit(np.concatenate(pooled))


def nearest_normal_score(
    descriptor: np.ndarray,
    normal_bank: np.ndarray,
    *,
    k: int = 3,
) -> tuple[float, list[dict[str, Any]]]:
    """Return mean cosine distance to the k closest normalized normal examples."""
    query = _unit(descriptor)
    bank = np.asarray(normal_bank, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[0] < 1 or bank.shape[1] != query.shape[0]:
        raise NormalReferenceError("normal bank shape does not match the descriptor")
    if k < 1 or k > bank.shape[0]:
        raise NormalReferenceError("k must be within the normal bank size")
    bank_norms = np.linalg.norm(bank, axis=1, keepdims=True)
    if np.any(~np.isfinite(bank_norms)) or np.any(bank_norms <= 1e-12):
        raise NormalReferenceError("normal bank contains an invalid descriptor")
    normalized_bank = bank / bank_norms
    distances = np.clip(1.0 - normalized_bank @ query, 0.0, 2.0)
    order = np.argsort(distances, kind="stable")[:k]
    neighbors = [
        {"index": int(index), "cosine_distance": float(distances[index])}
        for index in order
    ]
    return float(np.mean(distances[order])), neighbors


def classification_metrics(
    scores: Sequence[float],
    labels: Sequence[bool],
    threshold: float,
) -> dict[str, Any]:
    if len(scores) != len(labels) or not scores:
        raise NormalReferenceError("scores and labels must be non-empty and equally sized")
    numeric = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    if np.any(~np.isfinite(numeric)) or not np.isfinite(threshold):
        raise NormalReferenceError("scores and threshold must be finite")
    predicted = numeric >= float(threshold)
    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    true_negative = int(np.count_nonzero(~predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    sensitivity = ratio(true_positive, true_positive + false_negative)
    specificity = ratio(true_negative, true_negative + false_positive)
    precision = ratio(true_positive, true_positive + false_positive)
    accuracy = ratio(true_positive + true_negative, len(truth))
    balanced = (
        (sensitivity + specificity) / 2.0
        if sensitivity is not None and specificity is not None
        else None
    )
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision is not None and sensitivity is not None and precision + sensitivity > 0
        else None
    )
    return {
        "threshold": float(threshold),
        "image_count": len(truth),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "f1": f1,
    }


def select_balanced_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    minimum_sensitivity: float = 0.90,
) -> dict[str, Any]:
    """Choose a validation-only threshold with a fault-sensitivity floor."""
    if not 0.0 <= minimum_sensitivity <= 1.0:
        raise NormalReferenceError("minimum_sensitivity must be within 0..1")
    numeric = sorted({float(score) for score in scores})
    if not numeric:
        raise NormalReferenceError("at least one score is required")
    epsilon = max(1e-12, (numeric[-1] - numeric[0]) * 1e-9)
    thresholds = [numeric[0] - epsilon]
    thresholds.extend((left + right) / 2.0 for left, right in zip(numeric, numeric[1:]))
    thresholds.append(numeric[-1] + epsilon)
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = classification_metrics(scores, labels, threshold)
        if metrics["sensitivity"] is not None and metrics["sensitivity"] >= minimum_sensitivity:
            candidates.append(metrics)
    if not candidates:
        raise NormalReferenceError("no threshold satisfies the requested sensitivity floor")
    best = max(
        candidates,
        key=lambda item: (
            float(item["balanced_accuracy"] or 0.0),
            float(item["specificity"] or 0.0),
            float(item["precision"] or 0.0),
            float(item["threshold"]),
        ),
    )
    return {
        "selection_split": "validation_only",
        "objective": "maximize balanced accuracy, then specificity and precision",
        "minimum_sensitivity": minimum_sensitivity,
        "threshold": best["threshold"],
        "metrics": best,
        "eligible_threshold_count": len(candidates),
    }
