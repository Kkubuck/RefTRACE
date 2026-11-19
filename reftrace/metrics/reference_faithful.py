from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .matched import evaluate_arrays, find_mask, load_binary_mask, load_probability


EPSILON = 1e-12
SIZE_BINS = ("tiny", "small", "medium", "large")


def harmonic_mean(first: float, second: float, epsilon: float = EPSILON) -> float:
    return float(2.0 * first * second / (first + second + epsilon))


def object_size_bin(area_ratio: float) -> str:
    if area_ratio < 0.005:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.08:
        return "medium"
    return "large"


def size_balanced_mean(values: dict[str, list[float]]) -> float:
    bin_means = [float(np.mean(values[name])) for name in SIZE_BINS if values[name]]
    return float(np.mean(bin_means))


def rejection_sample(
    positive: np.ndarray,
    negative: np.ndarray,
    target: np.ndarray,
    tau_image: float = 0.05,
    tau_object: float = 0.05,
    tau_positive_object: float = 0.10,
) -> dict[str, float]:
    target_sum = float(target.sum())
    negative_image_activation = float(negative.mean())
    negative_object_activation = float((negative * target).sum() / (target_sum + EPSILON))
    blank_image = math.exp(-negative_image_activation / tau_image)
    blank_object = math.exp(-negative_object_activation / tau_object)
    blankness = harmonic_mean(blank_image, blank_object)
    positive_area = float(positive.mean())
    positive_object = float((positive * target).sum() / (target_sum + EPSILON))
    overlap = float(np.minimum(positive, negative).mean())
    switch_raw = float(np.clip(1.0 - overlap / (positive_area + EPSILON), 0.0, 1.0))
    positive_gate = float(np.clip(positive_object / tau_positive_object, 0.0, 1.0))
    switch = switch_raw * positive_gate
    return {
        "blankness": blankness,
        "switch": switch,
        "rrq": harmonic_mean(blankness, switch),
        "negative_image_activation": negative_image_activation,
        "negative_object_activation": negative_object_activation,
    }


def evaluate_reference_faithful_predictions(
    pred_root: str | Path,
    gt_root: str | Path,
    triplets: str | Path,
    tau_image: float = 0.05,
    tau_object: float = 0.05,
    tau_positive_object: float = 0.10,
) -> dict[str, float]:
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)
    with Path(triplets).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matched_pairs = []
    rejection = {
        "W": {name: [] for name in SIZE_BINS},
        "C": {name: [] for name in SIZE_BINS},
    }
    diagnostics = {"W_image": [], "W_object": [], "C_image": [], "C_object": []}
    for row in rows:
        stem = row["query_stem"]
        class_name = row["query_class"]
        target = load_binary_mask(find_mask(gt_root, class_name, stem))
        positive = load_probability(pred_root / "P" / f"{stem}.png", target.shape)
        matched_pairs.append((positive, target))
        size_bin = object_size_bin(float(target.mean()))
        for condition in ("W", "C"):
            negative = load_probability(pred_root / condition / f"{stem}.png", target.shape)
            score = rejection_sample(
                positive,
                negative,
                target,
                tau_image=tau_image,
                tau_object=tau_object,
                tau_positive_object=tau_positive_object,
            )
            rejection[condition][size_bin].append(score["rrq"])
            diagnostics[f"{condition}_image"].append(score["negative_image_activation"])
            diagnostics[f"{condition}_object"].append(score["negative_object_activation"])
    positive_metrics = evaluate_arrays(matched_pairs)
    rrq_w = size_balanced_mean(rejection["W"])
    rrq_c = size_balanced_mean(rejection["C"])
    rrq = harmonic_mean(rrq_w, rrq_c)
    return {
        **positive_metrics,
        "RRQ": rrq,
        "RRQ_W": rrq_w,
        "RRQ_C": rrq_c,
        "RFQ": harmonic_mean(positive_metrics["PSQ"], rrq),
        "W_image_activation": float(np.mean(diagnostics["W_image"])),
        "W_object_activation": float(np.mean(diagnostics["W_object"])),
        "C_image_activation": float(np.mean(diagnostics["C_image"])),
        "C_object_activation": float(np.mean(diagnostics["C_object"])),
        "samples": len(rows),
    }

