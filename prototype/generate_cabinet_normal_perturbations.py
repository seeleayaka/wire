"""Create deterministic, structure-preserving normal-image smoke fixtures.

The transforms in this module are deliberately global imaging effects only.
They never rotate, crop, warp, inpaint, add, remove, or relocate cabinet
content.  These fixtures are useful for checking false positives caused by
ordinary capture variation; they are not field-validation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VARIANT_DESCRIPTIONS = {
    "normal_dark": "global exposure multiplier 0.96",
    "normal_bright": "global exposure multiplier 1.04",
    "normal_warm": "global BGR white-balance multipliers 0.985, 1.000, 1.015",
    "normal_sensor_noise": "zero-mean sensor-like Gaussian noise, sigma 1.2",
    "normal_slight_blur": "global Gaussian blur, sigma 0.45",
    "normal_jpeg_texture": "JPEG encode/decode simulation, quality 97",
}


def read_image(path: Path) -> np.ndarray:
    """Read a Windows path safely, including Chinese characters."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write a PNG safely, including to a Windows path with Chinese characters."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    encoded.tofile(str(path))


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def generate_variants(reference: np.ndarray, *, seed: int = 20260826) -> dict[str, np.ndarray]:
    """Return six same-shape global-photometric variants of one BGR image."""
    if reference.ndim != 3 or reference.shape[2] != 3 or reference.dtype != np.uint8:
        raise ValueError("reference must be a uint8 BGR image")

    float_reference = reference.astype(np.float32)
    warm = float_reference.copy()
    warm[:, :, 0] *= 0.985
    warm[:, :, 2] *= 1.015
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.2, size=reference.shape).astype(np.float32)
    ok, encoded = cv2.imencode(".jpg", reference, [cv2.IMWRITE_JPEG_QUALITY, 97])
    if not ok:
        raise ValueError("cannot create JPEG texture simulation")
    jpeg_texture = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if jpeg_texture is None:
        raise ValueError("cannot decode JPEG texture simulation")

    variants = {
        "normal_dark": _uint8(float_reference * 0.96),
        "normal_bright": _uint8(float_reference * 1.04),
        "normal_warm": _uint8(warm),
        "normal_sensor_noise": _uint8(float_reference + noise),
        "normal_slight_blur": cv2.GaussianBlur(reference, (3, 3), 0.45),
        "normal_jpeg_texture": jpeg_texture,
    }
    for name, image in variants.items():
        if image.shape != reference.shape or image.dtype != np.uint8:
            raise AssertionError(f"{name} did not preserve image geometry and type")
    return variants


def create_fixture_set(reference_path: Path, output_dir: Path, *, seed: int = 20260826) -> dict[str, Any]:
    """Write a fresh fixture directory and its provenance metadata."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    reference = read_image(reference_path)
    variants = generate_variants(reference, seed=seed)
    output_dir.mkdir(parents=True)
    paths: dict[str, str] = {}
    for name, image in variants.items():
        target = output_dir / f"{name}.png"
        write_image(target, image)
        paths[name] = str(target)
    report = {
        "reference": str(reference_path),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "image_size": [int(reference.shape[1]), int(reference.shape[0])],
        "seed": seed,
        "fixture_kind": "synthetic_normal_photometric_smoke",
        "evidence_boundary": "These are global-photometric smoke fixtures, not field images or field accuracy evidence.",
        "forbidden_operations": ["AI_regeneration", "inpainting", "crop", "rotation", "perspective_warp", "local_content_edit"],
        "variants": [
            {"id": name, "description": VARIANT_DESCRIPTIONS[name], "path": paths[name]}
            for name in variants
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    report = create_fixture_set(args.reference, args.output, seed=args.seed)
    print(json.dumps({"output": str(args.output), "variant_count": len(report["variants"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
