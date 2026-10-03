"""Fixed entry point for the no-marker template inspection UI.

The shared RoiCanvas control exposes ``roi()``, while the first template UI
revision accidentally called it ``normalized_roi()``.  Keep the original
module intact and add this compatibility alias before importing it, so the
existing UI and saved recipes continue to work.
"""
from __future__ import annotations

from dimm_review_app_fixed import RoiCanvas


if not hasattr(RoiCanvas, "normalized_roi"):
    RoiCanvas.normalized_roi = RoiCanvas.roi  # type: ignore[attr-defined]


from assembly_template_review_app import main


if __name__ == "__main__":
    main()
