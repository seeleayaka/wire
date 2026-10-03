"""Windows-safe DINO review entry point.

PyTorch must load before PyQt on this machine; otherwise Qt's DLLs prevent
``c10.dll`` from initialising.  Keep this small wrapper separate from the visual
review implementation so the loading constraint remains explicit.
"""
from __future__ import annotations

# Load torch through the feature module before any import path reaches PyQt.
from dino_feature_diff_v2 import fused_components as _fused_components  # noqa: F401

import assembly_auto_review_dino as implementation


dino_fused_regions = implementation.dino_fused_regions
DINOReview = implementation.DINOReview


def main() -> None:
    implementation.main()


if __name__ == "__main__":
    main()
