"""Conservative review grouping; never fuse geometry or claim cable identity."""
from itertools import combinations
import math
import numpy as np


def group_segment_evidence(records, masks, *, minimum_iou=.9, endpoint_fraction=.05):
    """Complete-link groups preserve every raw record and original confidence.

    Each pair must share >=.9 mask IoU and agree on both endpoints within .05
    of their mask bbox diagonal. These are fixed review-group gates, not an
    accuracy calibration. Input order cannot affect grouping. No cross-tile
    stitching, confidence accumulation, geometry averaging or edge emission.
    """
    if not 0 < minimum_iou <= 1 or not 0 <= endpoint_fraction <= 1:
        raise ValueError("invalid grouping gates")
    ids, shapes = set(), set()
    by_id = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("record must be an object")
        rid = item.get("record_id")
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError("record IDs must be unique nonempty strings")
        ids.add(rid)
        if rid not in masks:
            raise ValueError("component mask missing")
        mask = np.asarray(masks[rid])
        if mask.ndim != 2 or not mask.size or not np.isfinite(mask).all():
            raise ValueError("mask must be finite 2D")
        mask = mask > 0
        if not mask.any():
            raise ValueError("component mask empty")
        shapes.add(mask.shape)
        score = item.get("source_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score must be finite within 0..1")
        by_id[rid] = (item, mask)
        if item.get("geometry_pair_eligible") is True:
            values = np.asarray(item.get("visible_ends_xy"), dtype=float)
            if values.shape != (2, 2) or not np.isfinite(values).all():
                raise ValueError("invalid endpoint coordinates")
            if not ((values[:, 0] >= 0).all() and (values[:, 0] < mask.shape[1]).all()
                    and (values[:, 1] >= 0).all() and (values[:, 1] < mask.shape[0]).all()):
                raise ValueError("endpoint coordinates outside mask frame")
    if len(shapes) > 1 or set(masks) != ids:
        raise ValueError("masks must share one frame and exactly match record IDs")

    def eligible(item):
        return (item.get("geometry_pair_eligible") is True and item.get("crop_boundary_guard_eligible") is not False
                and item.get("candidate_tip_count") == 2 and item.get("branch_cluster_count") == 0
                and item.get("skeleton_component_count") == 1
                and isinstance(item.get("visible_ends_xy"), list) and len(item["visible_ends_xy"]) == 2)

    pair_audit, compatible = [], {}
    for first, second in combinations(sorted(ids), 2):
        a, am = by_id[first]; b, bm = by_id[second]
        if not eligible(a) or not eligible(b):
            compatible[first, second] = False
            continue
        inter = int((am & bm).sum()); union = int((am | bm).sum())
        iou = inter / union
        end_distance = None
        if iou >= minimum_iou:
            def points(item):
                values = np.asarray(item["visible_ends_xy"], dtype=float)
                if values.shape != (2,2) or not np.isfinite(values).all():
                    raise ValueError("invalid endpoint coordinates")
                if not ((values[:,0] >= 0).all() and (values[:,0] < am.shape[1]).all()
                        and (values[:,1] >= 0).all() and (values[:,1] < am.shape[0]).all()):
                    raise ValueError("endpoint coordinates outside mask frame")
                return values
            av, bv = points(a), points(b)
            end_distance = min(float(np.linalg.norm(av-bv, axis=1).max()), float(np.linalg.norm(av-bv[::-1], axis=1).max()))
            ys, xs = np.where(am | bm)
            diagonal = math.hypot(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
            agrees = end_distance <= endpoint_fraction*diagonal
            pair_audit.append({"first": first, "second": second, "mask_iou": iou,
                               "max_paired_endpoint_distance": end_distance, "normalized_gate": endpoint_fraction,
                               "compatible": agrees})
        else:
            agrees = False
        compatible[first, second] = agrees
    groups = []
    for rid in sorted(ids):
        for group in groups:
            if all(compatible.get(tuple(sorted((rid, member))), False) for member in group):
                group.append(rid)
                break
        else:
            groups.append([rid])
    summaries = []
    for index, members in enumerate(groups, 1):
        representative = min(members, key=lambda name: (-by_id[name][0]["source_score"], name))
        summaries.append({"group_id": f"review_group_{index:03d}", "member_record_ids": members,
            "representative_record_id": representative,
            "representative_original_score": by_id[representative][0]["source_score"],
            "state": "near_duplicate_review_evidence" if len(members)>1 else "single_review_evidence",
            "confirmed_cable_identity": False})
    return {"schema_version": 1, "record_count": len(records), "review_group_count": len(groups),
        "duplicate_group_count": sum(len(g)>1 for g in groups),
        "display_redundancy_reduction": len(records)-len(groups), "groups": summaries,
        "pair_audit": pair_audit, "minimum_iou": minimum_iou, "endpoint_fraction": endpoint_fraction,
        "geometry_fused": False, "automatic_connections_emitted": False,
        "all_original_evidence_preserved": True}
