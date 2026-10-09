"""One frozen shape descriptor experiment, visible-body proposals only."""
import cv2
import numpy as np
from skimage.feature import hog

POLICY = {'implementation': 'existing_skimage_HOG_L2_Hys', 'window': [96,48], 'block': [16,16], 'stride': [8,8],
          'cell': [8,8], 'bins': 9, 'outer_offsets': [-2,0,2],
          'class_tail_alpha': .05, 'visible_only': True,
          'all_nine_HOG_and_color_singleton_1_required': True,
          'independent_observers': 1, 'electrical_continuity': 'not_assessed'}


def descriptor(rgb):
    rgb = np.asarray(rgb)
    if rgb.shape != (50,100,3) or rgb.dtype != np.uint8:
        raise ValueError('actual registered 100x50 uint8 RGB patch required')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    view = cv2.resize(gray, (96,48), interpolation=cv2.INTER_AREA)
    f = hog(view, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
            block_norm='L2-Hys', transform_sqrt=False, feature_vector=True).astype(np.float64)
    if f.shape != (1980,) or not np.isfinite(f).all():
        raise ValueError('finite1980 HOG dimensions required')
    return f


def resolve(previous, shape_labels, color_labels):
    for labels in [shape_labels, color_labels]:
        if len(labels) != 9 or any(v not in [None,0,1] or isinstance(v,bool) for v in labels):
            raise ValueError('nine explicit appearance labels required')
    if previous not in [None,0,1] or isinstance(previous,bool):
        raise ValueError('explicit prior appearance label required')
    return previous if previous is not None else (1 if all(v==1 for v in shape_labels+color_labels) else None)
