"""Fixed 480-source-pixel acquisition windows; detector input stays960."""
import math


def windows(seed, shape):
    h, w = shape
    l, t, r, b = map(float, seed['box_xyxy'])
    if min(h, w) < 480 or not all(math.isfinite(v) for v in (l, t, r, b)):
        raise ValueError('Unsupported 2x evidence geometry')
    if not (0 <= l < r <= w and 0 <= t < b <= h):
        raise ValueError('Evidence seed outside source')
    cx, cy = (l + r) / 2, (t + b) / 2
    return [[x, y, x + 480, y + 480] for dx, dy in ((-60, -60), (60, 60))
            for x, y in [(max(0, min(w - 480, round(cx + dx - 240))),
                          max(0, min(h - 480, round(cy + dy - 240))))]]
