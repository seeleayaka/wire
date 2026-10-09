"""A replacement detector occupies ONE role. GT-free provenance gates."""
from copy import deepcopy
import math


def validate_views(views, roles, source_sha, shape):
    if set(roles) != {'teacher', 'student', 'feature'} or len(set(roles.values())) != 3:
        raise ValueError('exactly three distinct detector roles required')
    if len(source_sha) != 64 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError('invalid source identity/frame')
    allowed = set(roles.values())
    seen = set()
    for view in views:
        digest = view['weight_sha256']
        if digest not in allowed:
            raise ValueError('old/replaced detector cannot vote for new cues')
        if view['source_sha256'] != source_sha or view['predictions']['source_shape'] != list(shape):
            raise ValueError('source/frame mismatch')
        for row in view['predictions']['merged_predictions']:
            box = row['box_xyxy']
            if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
                raise ValueError('nonfinite detector geometry')
            if row['class_id'] not in (0, 1) or isinstance(row['class_id'], bool):
                raise ValueError('invalid detector class')
            if not math.isfinite(float(row['confidence'])) or not 0 <= row['confidence'] <= 1:
                raise ValueError('invalid detector confidence')
        seen.add(digest)
    if seen != allowed:
        raise ValueError('missing detector role')
    return deepcopy(views)


def check_prefix(original, research, trial):
    """Exact protected rows, not just counts or approximate coordinates."""
    old = original['all_predictions']
    current = research['all_predictions']
    final = trial['all_predictions']
    if current[:len(old)] != old or final[:len(current)] != current:
        raise ValueError('protected output prefix changed')
    if len(original['primary']) > 5 or research['primary'] != original['primary'] or trial['primary'] != original['primary']:
        raise ValueError('protected primary cues changed')
    if len(final) > len(trial['primary']) + 5:
        raise ValueError('shared five-extra budget exceeded')
    return True
