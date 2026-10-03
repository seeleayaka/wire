"""Corrected SIFT fallback entry point for template review."""
from __future__ import annotations

import assembly_template_review_app as base

# Keep the original ORB function before loading the SIFT helper.  The helper's
# first revision replaces base.locate during import, so this saved reference is
# intentionally used by robust_locate below.
_orb_locate = base.locate
from assembly_template_review_app_sift import sift_fallback  # noqa: E402


def robust_locate(reference, inspection, template_roi):  # type: ignore[no-untyped-def]
    aligned, report = _orb_locate(reference, inspection, template_roi)
    if aligned is not None:
        return aligned, report
    fallback_aligned, fallback_report = sift_fallback(reference, inspection, template_roi)
    fallback_report["orb_first_attempt"] = report
    return fallback_aligned, fallback_report


base.locate = robust_locate


if __name__ == "__main__":
    base.main()
