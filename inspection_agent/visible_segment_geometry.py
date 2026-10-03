"""Conservative visible-skeleton evidence, not physical cable connectivity."""
from __future__ import annotations

import cv2
import numpy as np


def assess_visible_skeleton(skeleton: np.ndarray) -> dict:
    """Keep every tip; never choose an arbitrary pair from a branched component.

    No spur pruning or scene-specific thresholds. Pixel-grid branch ambiguity
    may conservatively reject a real simple wire; rejection is not a defect.
    A two-ended visible segment still proves no hidden/electrical connection.
    """
    raw = np.asarray(skeleton)
    if raw.ndim != 2 or not raw.size:
        raise ValueError("skeleton must be a nonempty 2D array")
    active = (raw > 0).astype(np.uint8)
    neighbors = cv2.filter2D(
        active, cv2.CV_16S, np.ones((3, 3), np.int16),
        borderType=cv2.BORDER_CONSTANT,
    ) - active
    ys, xs = np.where((active > 0) & (neighbors == 1))
    tips = [[int(x), int(y)] for x, y in zip(xs, ys)]
    branches = ((active > 0) & (neighbors >= 3)).astype(np.uint8)
    branch_clusters = int(cv2.connectedComponents(branches, connectivity=8)[0] - 1)
    component_count = int(cv2.connectedComponents(active, connectivity=8)[0] - 1)
    if component_count != 1:
        status = "ambiguous_disconnected_or_empty_skeleton"
    elif branch_clusters:
        status = "ambiguous_branched_visible_segment"
    elif len(tips) != 2:
        status = "ambiguous_visible_segment_end_count"
    else:
        status = "unbranched_two_tip_visible_segment"
    accepted = status == "unbranched_two_tip_visible_segment"
    return {
        "geometry_contract_version": 1,
        "skeleton_component_count": component_count,
        "skeleton_pixels": int(active.sum()),
        "candidate_tip_count": len(tips),
        "candidate_tips_xy": tips,
        "branch_cluster_count": branch_clusters,
        "geometry_pair_eligible": accepted,
        "visible_ends_xy": tips if accepted else [],
        "endpoint_status": status,
    }
