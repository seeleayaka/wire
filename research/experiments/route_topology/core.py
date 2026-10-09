"""Optional, scoped visible-route review. NOT electrical continuity.

Independent of the production detector. No fragment bridging, spur pruning,
nearest-port guessing, cross-view coordinate reuse or threshold sweep.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_binding(path):
    path = Path(path)
    with Image.open(path) as image:
        size = list(image.size)
        image.verify()
    return {"image_sha256": sha256(path), "image_size": size,
            "coordinate_frame": "source_image_pixels"}


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + " must be nonempty text")
    return value.strip()


def reviewed(value):
    if not isinstance(value, dict) or type(value.get("confirmed")) is not bool:
        raise ValueError("review must contain an explicit confirmed boolean")
    if value["confirmed"]:
        text(value.get("reviewer"), "reviewer")
        text(value.get("evidence_note"), "evidence_note")
    return value["confirmed"]


def validate_ports(ports, size):
    if not isinstance(ports, list):
        raise ValueError("ports must be a list")
    seen = set()
    for port in ports:
        identity = text(port.get("id"), "port ID")
        if identity in seen:
            raise ValueError("duplicate port identity")
        seen.add(identity)
        if port.get("roi_kind") != "wire_entry_port":
            raise ValueError("body boxes are not wire-entry ports")
        box = port.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box
        ):
            raise ValueError("ROI requires four finite numbers")
        a, b, c, d = box
        if not (0 <= a < c <= size[0] and 0 <= b < d <= size[1]):
            raise ValueError("port ROI outside the bound image")
        reviewed(port)
    return seen


def trace_skeleton(skeleton):
    """A digital path, with redundant corner diagonals excluded, never pruned.

    Remove a diagonal adjacency ONLY if an occupied orthogonal intermediate
    already connects its endpoints. This changes no foreground components,
    deletes no pixels, and never joins gaps. True branches/cycles still reject.
    A valid digital path is not certification of a physical cable.
    """
    raw = np.asarray(skeleton)
    if raw.ndim != 2 or not raw.size or not np.issubdtype(raw.dtype, np.number) and raw.dtype != bool:
        raise ValueError("skeleton must be a nonempty numeric 2D array")
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite skeleton")
    pixels = {tuple(map(int, p)) for p in np.argwhere(raw > 0)}
    adjacency = {p: [] for p in pixels}
    removed = 0
    for y, x in sorted(pixels):
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            q = (y + dy, x + dx)
            if q not in pixels:
                continue
            if dy and dx and ((y, x + dx) in pixels or (y + dy, x) in pixels):
                removed += 1
                continue
            adjacency[(y, x)].append(q)
            adjacency[q].append((y, x))
    remaining = set(pixels)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            for q in adjacency[stack.pop()]:
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
    tips = sorted(p for p in pixels if len(adjacency[p]) == 1)
    branches = sorted(p for p in pixels if len(adjacency[p]) > 2)
    state = ("disconnected_or_empty" if components != 1 else
             "branched_or_crossing" if branches else
             "not_two_ended" if len(tips) != 2 else "simple_visible_path")
    result = {"geometry_contract_version": 2, "state": state,
              "component_count": components, "tip_count": len(tips),
              "branch_pixel_count": len(branches), "skeleton_pixels": len(pixels),
              "redundant_diagonal_edges_removed": removed,
              "tips_xy": [[x, y] for y, x in tips], "path_xy": []}
    if state == "simple_visible_path":
        previous, current = None, tips[0]
        path = []
        while True:
            path.append([current[1], current[0]])
            successors = [q for q in adjacency[current] if q != previous]
            if not successors:
                break
            previous, current = current, successors[0]
        if len(path) != len(pixels) or current != tips[1]:
            raise AssertionError("path traversal did not cover exactly all pixels")
        result["path_xy"] = path
    return result


def extract_mask(mask, score, record_id):
    raw = np.asarray(mask)
    if raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError("mask must be a finite nonempty 2D array")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("score must be finite within 0..1")
    # Treat each original instance as a whole: disconnected fragments do not
    # become independent fully-connected cables by splitting and choosing one.
    geometry = trace_skeleton(skeletonize(raw > 0))
    h, w = raw.shape
    ys, xs = np.where(raw > 0)
    touches = bool(len(xs) and (xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2))
    return {"record_id": text(record_id, "record ID"), "score": score,
            "mask_array_sha256": hashlib.sha256((raw > 0).astype(np.uint8).tobytes()).hexdigest(),
            "boundary_truncated": touches, "geometry": geometry}


def matches(point, ports):
    x, y = point
    return sorted(p["id"] for p in ports if
                  p["bbox_xyxy"][0] <= x <= p["bbox_xyxy"][2] and
                  p["bbox_xyxy"][1] <= y <= p["bbox_xyxy"][3])


def assess_view(records, ports, size, coverage_review):
    validate_ports(ports, size)
    coverage = reviewed(coverage_review)
    reasons = []
    if not coverage:
        reasons.append("visible_scope_coverage_not_reviewed")
    if not ports or any(not p["confirmed"] for p in ports):
        reasons.append("port_identity_not_confirmed")
    if not records:
        reasons.append("no_mask_evidence")
    edges, audit = defaultdict(list), []
    ids = set()
    for record in records:
        rid = text(record.get("record_id"), "record ID")
        if rid in ids:
            raise ValueError("duplicate record ID")
        ids.add(rid)
        score = record.get("score")
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid record score")
        g = record["geometry"]
        # These records must be produced by extract_mask in the same invocation,
        # not arbitrary JSON endpoint declarations (enforced by the CLI).
        state = g["state"]
        endpoint_hits = []
        if state == "simple_visible_path":
            if g.get("geometry_contract_version") != 2 or len(g["tips_xy"]) != 2 or not g["path_xy"]:
                raise ValueError("geometry contract invalid")
            if any(not (0 <= x < size[0] and 0 <= y < size[1]) for x, y in g["path_xy"]):
                raise ValueError("path outside bound image")
            endpoint_hits = [matches(p, ports) for p in g["tips_xy"]]
            if record["boundary_truncated"]:
                state = "boundary_truncated"
            elif score < .75:
                state = "low_segmentation_score"
            elif "port_identity_not_confirmed" in reasons:
                state = "port_identity_not_confirmed"
            elif any(len(m) != 1 for m in endpoint_hits):
                state = "endpoint_unassigned_or_ambiguous"
            elif endpoint_hits[0] == endpoint_hits[1]:
                state = "both_ends_same_port"
            else:
                edge = tuple(sorted((endpoint_hits[0][0], endpoint_hits[1][0])))
                touched = {identity for p in g["path_xy"] for identity in matches(p, ports)}
                if touched != set(edge):
                    state = "path_intersects_other_port_roi"
                else:
                    state = "candidate_visible_relation"
                    edges[edge].append(rid)
        audit.append({"record_id": rid, "state": state, "endpoint_hits": endpoint_hits,
                      "score": score, "geometry": deepcopy(g)})
    if any(item["state"] != "candidate_visible_relation" for item in audit):
        reasons.append("unresolved_selected_mask")
    # A selected mask must not yield competing wires sharing a port. Do not
    # pick a highest-scoring relation from contradictory segmentation instances.
    incidence = defaultdict(set)
    for edge in edges:
        for identity in edge:
            incidence[identity].add(edge)
    if any(len(items) > 1 for items in incidence.values()):
        reasons.append("conflicting_relations_at_port")
    relation_paths = []
    for item in audit:
        if item["state"] == "candidate_visible_relation":
            key = tuple(sorted(h[0] for h in item["endpoint_hits"]))
            relation_paths.append((key, {tuple(p) for p in item["geometry"]["path_xy"]}))
    for i, (key, path) in enumerate(relation_paths):
        for other_key, other_path in relation_paths[i + 1:]:
            if key != other_key and path & other_path:
                reasons.append("overlapping_different_relation_paths")
                break
    reasons = list(dict.fromkeys(reasons))
    return {"reasons": reasons, "evidence": audit,
            "edges": [{"from": a, "to": b, "evidence_ids": evidence}
                      for (a, b), evidence in sorted(edges.items())],
            "complete_visible_scope": not reasons}


def compare_views(reference, inspection, expected_edges, expected_review):
    confirmed = reviewed(expected_review)
    expected = None
    if expected_edges is not None:
        if not isinstance(expected_edges, list):
            raise ValueError("expected edges must be null or a list")
        expected = set()
        for edge in expected_edges:
            a, b = text(edge.get("from"), "from"), text(edge.get("to"), "to")
            key = tuple(sorted((a, b)))
            if a == b or key in expected:
                raise ValueError("self or duplicate expected edge")
            expected.add(key)
    elif confirmed:
        raise ValueError("unknown expectation cannot be confirmed")
    ref = {tuple(sorted((e["from"], e["to"]))) for e in reference["edges"]}
    ins = {tuple(sorted((e["from"], e["to"]))) for e in inspection["edges"]}
    reasons = ["reference:" + r for r in reference["reasons"]] + ["inspection:" + r for r in inspection["reasons"]]
    if expected is None or not confirmed:
        reasons.append("expected_connections_unknown_or_unreviewed")
    if expected is not None and ref != expected:
        reasons.append("reference_does_not_establish_expected_connections")
    state = "insufficient_evidence" if reasons else (
        "same_visible_terminal_relations" if ref == ins else "visible_terminal_relation_difference")
    # Geometry never votes on fault correctness; shapes can change under camera
    # motion. Geometry is deliberately not used to prove that routing changed.
    return {"schema_version": 1, "decision": state, "reasons": reasons,
            "missing_visible_relations": [list(e) for e in sorted(ref - ins)],
            "unexpected_visible_relations": [list(e) for e in sorted(ins - ref)],
            "reference": reference, "inspection": inspection,
            "visual_difference_may_be_route_only": state == "same_visible_terminal_relations",
            "automatic_fault_verdict": False, "electrical_continuity": "not_assessed",
            "manual_confirmation_required": True,
            "claim_boundary": "Selected visible segments only; no whole cabinet, hidden path, cable identity, seating or continuity verdict."}
