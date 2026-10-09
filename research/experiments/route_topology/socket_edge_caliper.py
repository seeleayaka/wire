"""Reference-only geometric appearance probe. NOT insertion-depth/continuity proof.

Frozen policy before source outputs. No source labels, fitting, mask closing,
pixel interpolation of gaps, or inspection-name rules. Similar static edges
can mimic a connector: even a pass cannot confirm a wire-to-port relation.
"""
import cv2
import numpy as np

POLICY = dict(canny_low=60, canny_high=120, distance_px=1.5,
              orientation_cos_min=.8, bilateral_support_min=.8,
              min_edges=25, translation_max=2, core_fraction=.8)


def edges(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (50, 100, 3):
        raise ValueError('registered 100x50 uint8 RGB required')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Photometric normalization only; never modifies geometric topology.
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    edge = cv2.Canny(gray, POLICY['canny_low'], POLICY['canny_high']) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    norm = np.maximum(np.sqrt(gx*gx+gy*gy), 1e-6)
    return edge, np.stack([gx/norm, gy/norm], axis=-1)


def directional_support(a, va, b, vb):
    ay, ax = np.nonzero(a); by, bx = np.nonzero(b)
    if len(ax) < POLICY['min_edges'] or len(bx) < POLICY['min_edges']:
        return 0.
    delta = np.stack([ay, ax], 1)[:, None] - np.stack([by, bx], 1)[None]
    distances = (delta*delta).sum(2)
    nearest = distances.argmin(1)
    cos = np.abs((va[ay, ax]*vb[by[nearest], bx[nearest]]).sum(1))
    return float(((distances[np.arange(len(ax)), nearest] <= POLICY['distance_px']**2)
                  & (cos >= POLICY['orientation_cos_min'])).mean())


def compare(reference, inspection):
    re, rv = edges(reference); ie, iv = edges(inspection)
    # Fixed interior excludes patch borders. Both directions avoid accepting a
    # few matching motherboard edges in a substantially missing connector.
    core = np.zeros_like(re); core[5:45, 10:90] = True
    candidates = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            aligned = np.roll(ie, (dy, dx), (0, 1)) & core
            directions = np.roll(iv, (dy, dx), (0, 1))
            forward = directional_support(re & core, rv, aligned, directions)
            reverse = directional_support(aligned, directions, re & core, rv)
            candidates.append((min(forward, reverse), -(abs(dx)+abs(dy)), dx, dy, forward, reverse))
    score, _, dx, dy, forward, reverse = max(candidates)
    return dict(reference_edge_count=int((re & core).sum()),
                inspection_edge_count=int((ie & core).sum()),
                forward_support=forward, reverse_support=reverse,
                bilateral_support=score, residual_translation=[dx, dy],
                reference_geometry_compatible=score >= POLICY['bilateral_support_min'],
                decision='insufficient_evidence', plug_seating_assessed=False,
                new_confirmed_connections=0, electrical_continuity='not_assessed')
