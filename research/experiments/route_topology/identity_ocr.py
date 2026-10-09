"""Local wire/terminal text nominations, never authenticated terminal IDs.

Fixed full-image 2x views at 0/90/180/270 degrees. Same weights count ONE
observer; no O/0, I/1 repair, nearest-port assignment, or automatic confirmation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

import numpy as np


def inverse_rotation(points, width, height, quarter_turns, scale):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('points require finite Nx2 coordinates')
    if type(quarter_turns) is not int or not 0 <= quarter_turns <= 3:
        raise ValueError('quarter_turns must be 0..3')
    if not math.isfinite(scale) or scale <= 0 or width <= 0 or height <= 0:
        raise ValueError('invalid image geometry')
    x, y = p[:, 0].copy(), p[:, 1].copy()
    if quarter_turns == 1:
        p[:, 0], p[:, 1] = width - 1 - y, x
    elif quarter_turns == 2:
        p[:, 0], p[:, 1] = width - 1 - x, height - 1 - y
    elif quarter_turns == 3:
        p[:, 0], p[:, 1] = y, height - 1 - x
    return (p / scale).tolist()


def box_iou(first, second):
    a, b = first, second
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.


def summarize_readings(readings):
    """Spatial/text diagnostic only. Never infer a port or edge from OCR boxes."""
    groups = []
    for row in sorted(deepcopy(readings), key=lambda r: (-r['score'], r['record_id'])):
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError('empty OCR text')
        score = row['score']
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid OCR score')
        box = row['bbox_xyxy']
        if len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('invalid OCR box')
        # Pairwise overlap, NOT transitive union, so a bridge box can't merge
        # two separate physical labels into one identity.
        candidates = [g for g in groups if all(box_iou(box, item['bbox_xyxy']) >= .5 for item in g)]
        if len(candidates) == 1:
            candidates[0].append(row)
        else:
            groups.append([row])
    output = []
    for i, group in enumerate(groups, 1):
        high = [r for r in group if r['score'] >= .9]
        variants = sorted({r['text'].strip() for r in group})
        high_texts = {r['text'].strip() for r in high}
        best = group[0]
        output.append({'group_id': f'text_{i:03}', 'bbox_xyxy': best['bbox_xyxy'],
            'text_variants': variants, 'best_text': best['text'].strip(),
            'best_score': best['score'], 'ocr_high_score_consistent': bool(high) and len(variants) == 1,
            'high_score_text_count': len(high_texts), 'model_observer_count': 1,
            'view_count': len({r['view_quarter_turns'] for r in group}),
            'simple_ascii_identity_candidate': bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9:_/.-]*', best['text'].strip())),
            'confirmed': False, 'terminal_assignment': None, 'wire_identity': None,
            'record_ids': [r['record_id'] for r in group]})
    repeated = {}
    for group in output:
        if group['ocr_high_score_consistent']:
            repeated.setdefault(group['best_text'], []).append(group['group_id'])
    return {'groups': output, 'repeated_text_groups': {t:g for t,g in repeated.items() if len(g) > 1},
            'automatic_connections': [], 'confirmed_port_identities': [],
            'claim_boundary': 'OCR text nominations only; repeated labels do not prove same conductor, port assignment, expected wiring or continuity.'}
