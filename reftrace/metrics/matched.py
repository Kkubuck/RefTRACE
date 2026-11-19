from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from py_sod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure


EPSILON = 1e-12


def harmonic_mean(values: list[float], epsilon: float = EPSILON) -> float:
    if not values:
        raise ValueError("harmonic_mean requires at least one value")
    return float(len(values) / sum(1.0 / max(float(value), epsilon) for value in values))


def _unit_interval(value: float) -> float:
    """Suppress metric-library round-off outside the mathematical [0, 1] range."""
    return float(np.clip(value, 0.0, 1.0))


def load_probability(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if shape is not None and image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    if array.max(initial=0.0) > 1.5:
        array /= 255.0
    return np.clip(array, 0.0, 1.0)


def load_binary_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return (array > 0.5).astype(np.float32)


def find_mask(gt_root: Path, class_name: str, stem: str) -> Path:
    directory = gt_root / class_name
    for extension in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No ground-truth mask for {class_name}/{stem}")


def evaluate_arrays(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    stores = {"S": Smeasure(), "wF": WeightedFmeasure(), "MAE": MAE(), "E": Emeasure()}
    for prediction, target in pairs:
        prediction_u8 = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)
        target_u8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)
        for metric in stores.values():
            metric.step(pred=prediction_u8, gt=target_u8)
    adaptive_e = stores["E"].get_results()["em"]["adp"]
    result = {
        "Smeasure": _unit_interval(stores["S"].get_results()["sm"]),
        "WeightedFmeasure": _unit_interval(stores["wF"].get_results()["wfm"]),
        "MAE": _unit_interval(stores["MAE"].get_results()["mae"]),
        "AdaptiveEmeasure": _unit_interval(adaptive_e),
    }
    result["PSQ"] = _unit_interval(
        harmonic_mean(
            [
                result["Smeasure"],
                result["WeightedFmeasure"],
                1.0 - result["MAE"],
                result["AdaptiveEmeasure"],
            ]
        )
    )
    return result


def evaluate_matched_predictions(pred_root: str | Path, gt_root: str | Path) -> dict[str, float]:
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)
    pairs = []
    for prediction_path in sorted(pred_root.glob("*.png")):
        matches = list(gt_root.glob(f"*/{prediction_path.stem}.*"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one GT match for {prediction_path.stem}, found {len(matches)}"
            )
        target = load_binary_mask(matches[0])
        prediction = load_probability(prediction_path, target.shape)
        pairs.append((prediction, target))
    if not pairs:
        raise RuntimeError(f"No PNG predictions found under {pred_root}")
    result = evaluate_arrays(pairs)
    result["samples"] = len(pairs)
    return result
