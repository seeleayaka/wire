"""DINOv2 feature-difference branch for aligned visual inspection images.

This module deliberately has no UI dependency.  It caches fixed-reference
features, produces a token-level DINO heatmap, and fuses it with conventional
colour/edge evidence.  A caller decides how to draw or threshold candidates.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import cv2
import numpy as np


MODEL_NAME = "facebookresearch/dinov2:dinov2_vits14"
PATCH_SIZE = 14
MAX_EDGE = 392
MIN_EDGE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# Candidate extraction later normalizes each ROI by its own percentiles.  Keep
# a small absolute-signal gate before that normalization: otherwise harmless
# global capture variation is stretched into a full-strength candidate map.
# These deliberately conservative floors are separated from the cabinet smoke
# fixtures (DINO P95 <= 0.002, traditional P95 <= 9) and the existing wrong
# fixtures (DINO P95 >= 0.178, traditional P95 = 204).  They are a smoke-test
# guard only, not a field-accuracy threshold.
LOW_SIGNAL_DINO_P95_MAX = 0.010
LOW_SIGNAL_TRADITIONAL_P95_MAX = 12.0
# Reference features are an expendable optimisation.  They must not make the
# operator UI fall back from DINO just because a project checkout is read-only.
# A deployment may point this to a durable writable directory when desired.
CACHE_DIR = Path(
    os.environ.get(
        "PYTHONPROJECT10_DINO_CACHE",
        str(Path(tempfile.gettempdir()) / "PythonProject10" / "dino_cache"),
    )
)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "dinov2"
MODEL_WEIGHTS = MODEL_DIR / "weights" / "dinov2_vits14_pretrain.pth"
_MODEL: torch.nn.Module | None = None


def _model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        if not MODEL_DIR.is_dir() or not MODEL_WEIGHTS.is_file():
            raise RuntimeError(
                "本地 DINOv2 模型文件缺失。需要 models/dinov2/ 和 "
                "models/dinov2/weights/dinov2_vits14_pretrain.pth。"
            )
        _MODEL = torch.hub.load(
            str(MODEL_DIR),
            "dinov2_vits14",
            source="local",
            weights=str(MODEL_WEIGHTS),
        )
        _MODEL.eval()
    return _MODEL


def _input_size(height: int, width: int) -> tuple[int, int]:
    """Keep aspect ratio while making both dimensions valid DINO patch multiples."""
    scale = MAX_EDGE / float(max(height, width))
    scaled_height, scaled_width = height * scale, width * scale
    if min(scaled_height, scaled_width) < MIN_EDGE:
        scale = MIN_EDGE / float(min(height, width))
        scaled_height, scaled_width = height * scale, width * scale
    output_height = max(PATCH_SIZE, int(round(scaled_height / PATCH_SIZE)) * PATCH_SIZE)
    output_width = max(PATCH_SIZE, int(round(scaled_width / PATCH_SIZE)) * PATCH_SIZE)
    return output_height, output_width


def _cache_key(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(MODEL_NAME.encode("utf-8"))
    digest.update(str(image.shape).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _preprocess(image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    target_height, target_width = _input_size(*image.shape[:2])
    rgb = cv2.cvtColor(cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0), (target_height, target_width)


def extract_features(image: np.ndarray, cache_reference: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Return normalized [grid_height, grid_width, channels] patch features."""
    key = _cache_key(image)
    cache_path = CACHE_DIR / f"{key}.npz"
    if cache_reference and cache_path.is_file():
        saved = np.load(cache_path)
        return saved["features"], {"cache": "hit", "input_height": int(saved["input_height"]), "input_width": int(saved["input_width"])}

    tensor, (input_height, input_width) = _preprocess(image)
    with torch.inference_mode():
        output = _model().forward_features(tensor)
    tokens = output["x_norm_patchtokens"][0].cpu().numpy().astype(np.float32)
    grid_height, grid_width = input_height // PATCH_SIZE, input_width // PATCH_SIZE
    features = tokens.reshape(grid_height, grid_width, -1)
    features /= np.maximum(np.linalg.norm(features, axis=2, keepdims=True), 1e-8)
    if cache_reference:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features, input_height=input_height, input_width=input_width)
    return features, {"cache": "miss" if cache_reference else "not_used", "input_height": input_height, "input_width": input_width}


def dino_difference(reference: np.ndarray, inspection: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Token-wise cosine distance, gently smoothed and upsampled to crop pixels."""
    reference_features, reference_meta = extract_features(reference, cache_reference=True)
    inspection_features, inspection_meta = extract_features(inspection, cache_reference=False)
    if reference_features.shape[:2] != inspection_features.shape[:2]:
        raise ValueError("reference and inspection feature grids differ; they must use the same crop geometry")
    similarity = np.sum(reference_features * inspection_features, axis=2)
    difference = cv2.GaussianBlur(np.clip(1.0 - similarity, 0.0, 2.0), (3, 3), 0)
    upsampled = cv2.resize(difference, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    return upsampled.astype(np.float32), {
        "model": MODEL_NAME,
        "patch_grid": [int(reference_features.shape[1]), int(reference_features.shape[0])],
        "reference_cache": reference_meta["cache"],
        "input_size": [reference_meta["input_width"], reference_meta["input_height"]],
        "dino_p50": round(float(np.percentile(upsampled, 50)), 5),
        "dino_p95": round(float(np.percentile(upsampled, 95)), 5),
    }


def traditional_difference(reference: np.ndarray, inspection: np.ndarray) -> np.ndarray:
    """Conventional colour/edge signal, kept separate for auditable fusion."""
    def gray(image: np.ndarray) -> np.ndarray:
        value = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX), (3, 3), 0)

    reference_gray, inspection_gray = gray(reference), gray(inspection)
    tone = cv2.absdiff(reference_gray, inspection_gray).astype(np.float32)
    edge = cv2.absdiff(cv2.Canny(reference_gray, 45, 120), cv2.Canny(inspection_gray, 45, 120)).astype(np.float32)
    reference_lab = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2LAB), (3, 3), 0)
    inspection_lab = cv2.GaussianBlur(cv2.cvtColor(inspection, cv2.COLOR_BGR2LAB), (3, 3), 0)
    chroma = cv2.absdiff(reference_lab[:, :, 1:], inspection_lab[:, :, 1:]).mean(axis=2).astype(np.float32)
    return np.maximum.reduce((tone, edge * 0.80, chroma * 1.35)).astype(np.float32)


def _roi_normalize(score: np.ndarray) -> np.ndarray:
    low, high = np.percentile(score, (50.0, 97.0))
    return np.clip((score - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def fused_evidence(
    reference: np.ndarray, inspection: np.ndarray, *, require_cross_evidence: bool = False
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Fuse both branches and retain their maps for candidate-level auditing."""
    if reference.shape != inspection.shape:
        raise ValueError(f"reference and inspection ROI shapes differ: {reference.shape} != {inspection.shape}")
    traditional = traditional_difference(reference, inspection)
    dino, metadata = dino_difference(reference, inspection)
    if traditional.shape != dino.shape:
        raise AssertionError(f"traditional and DINO score shapes differ: {traditional.shape} != {dino.shape}")
    traditional_p95 = float(np.percentile(traditional, 95))
    dino_p95 = float(np.percentile(dino, 95))
    low_absolute_signal = (
        traditional_p95 <= LOW_SIGNAL_TRADITIONAL_P95_MAX
        and dino_p95 <= LOW_SIGNAL_DINO_P95_MAX
    )
    metadata.update(
        traditional_p95=round(traditional_p95, 3),
        absolute_signal_gate=("suppressed_low_signal" if low_absolute_signal else "passed"),
        absolute_signal_gate_limits={
            "dino_p95_max": LOW_SIGNAL_DINO_P95_MAX,
            "traditional_p95_max": LOW_SIGNAL_TRADITIONAL_P95_MAX,
        },
    )
    if low_absolute_signal:
        # Do not percentile-normalize near-zero residuals.  The caller still
        # receives auditable raw maps and an explicit suppression reason.
        empty = np.zeros_like(traditional, dtype=np.float32)
        metadata.update(
            dino_p50=round(float(np.percentile(dino, 50)), 5),
            dino_p95=round(dino_p95, 5),
            fused_p95=0.0,
            agreement_p95=0.0,
            fusion_policy="suppressed_low_absolute_signal",
            score_units="0_to_255_fused_pixel_score",
            input_size_units="pixels_width_height",
            patch_grid_units="DINO_patches_width_height",
        )
        return empty, metadata, {
            "traditional": traditional,
            "dino": dino,
            "traditional_normalized": empty,
            "dino_normalized": empty,
            "agreement": empty,
        }
    traditional_normal = _roi_normalize(traditional)
    dino_normal = _roi_normalize(dino)
    agreement = np.sqrt(traditional_normal * dino_normal)
    if require_cross_evidence:
        # Keep thin wire changes despite DINO's coarse patches, but do not allow
        # a conventional rail/label edge to produce a cabinet-wide candidate alone.
        fused = np.maximum(agreement, np.minimum(traditional_normal, dino_normal + 0.24))
        fusion_policy = "cross_evidence_large_review"
    else:
        fused = np.maximum(np.maximum(traditional_normal * 0.84, dino_normal * 0.84), agreement)
        fusion_policy = "sensitive_targeted_review"
    metadata.update(
        fused_p95=round(float(np.percentile(fused, 95)), 5),
        agreement_p95=round(float(np.percentile(agreement, 95)), 5),
        fusion_policy=fusion_policy,
        score_units="0_to_255_fused_pixel_score",
        input_size_units="pixels_width_height",
        patch_grid_units="DINO_patches_width_height",
    )
    return (fused * 255.0).astype(np.float32), metadata, {
        "traditional": traditional,
        "dino": dino,
        "traditional_normalized": traditional_normal,
        "dino_normalized": dino_normal,
        "agreement": agreement,
    }


def fused_difference(
    reference: np.ndarray, inspection: np.ndarray, *, require_cross_evidence: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse conventional and DINO evidence for targeted or whole-cabinet review."""
    score, metadata, _evidence = fused_evidence(
        reference,
        inspection,
        require_cross_evidence=require_cross_evidence,
    )
    return score, metadata
