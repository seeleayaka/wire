"""Verify that locally provisioned DINOv2 and LightGlue assets can be imported."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DINO_DIR = ROOT / "models" / "dinov2"
DINO_WEIGHTS = DINO_DIR / "weights" / "dinov2_vits14_pretrain.pth"
LIGHTGLUE_DIR = ROOT / "models" / "LightGlue"
LIGHTGLUE_WEIGHTS = LIGHTGLUE_DIR / "lightglue" / "weights" / "sift_lightglue_v0-1_arxiv.pth"


def main() -> None:
    missing = [str(path) for path in (DINO_DIR, DINO_WEIGHTS, LIGHTGLUE_DIR, LIGHTGLUE_WEIGHTS) if not path.exists()]
    if missing:
        raise SystemExit("local model assets missing: " + ", ".join(missing))

    import torch

    model = torch.hub.load(str(DINO_DIR), "dinov2_vits14", source="local", weights=str(DINO_WEIGHTS))
    model.eval()
    sys.path.insert(0, str(LIGHTGLUE_DIR))
    from lightglue import LightGlue, SIFT  # noqa: F401

    matcher = LightGlue(
        features=None,
        input_dim=128,
        add_scale_ori=True,
        weights="sift_lightglue_v0-1_arxiv",
    ).eval()
    del matcher

    print("local DINOv2 checkpoint: OK")
    print("local LightGlue SIFT matcher and weights: OK")


if __name__ == "__main__":
    main()
