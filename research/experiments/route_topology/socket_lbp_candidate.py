"""Fixed local binary texture cue; one visual observer, never connectivity."""
import numpy as np
from skimage.feature import local_binary_pattern

POLICY = {'kind': 'source_only_socket_texture', 'grid': [4, 2], 'points': 8,
          'radii': [1, 2], 'method': 'uniform', 'offsets': [-2, 0, 2],
          'class_tail_alpha': .05, 'model_observer_count': 1,
          'electrical_continuity': 'not_assessed'}

def descriptor(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (50, 100, 3):
        raise ValueError('registered native 100x50 RGB patch required')
    gray = np.rint(rgb.astype(float) @ np.array([.299, .587, .114])).astype(np.uint8)
    values = []
    for radius in POLICY['radii']:
        codes = local_binary_pattern(gray, 8, radius, method='uniform')
        for y in range(2):
            for x in range(4):
                region = codes[y*25:(y+1)*25, x*25:(x+1)*25]
                values.extend(np.bincount(region.astype(int).ravel(), minlength=10) / region.size)
    result = np.array(values, dtype=np.float64)
    if result.shape != (160,) or not np.isfinite(result).all():
        raise ValueError('invalid texture descriptor')
    return result

def resolve(previous, labels):
    if previous not in (None, 0, 1):
        raise ValueError('invalid previous visual label')
    if len(labels) != 9 or any(v not in (None, 0, 1) for v in labels):
        raise ValueError('nine fixed-offset labels required')
    if previous is not None:
        return previous
    return labels[0] if labels[0] is not None and all(v == labels[0] for v in labels) else None
