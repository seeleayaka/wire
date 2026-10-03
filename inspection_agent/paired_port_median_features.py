"""GT-free equal-distinct-checkpoint geometry; independent of source IDs."""
import copy
import math
import statistics
from inspection_agent.port_tiling import box_iou


def proposals(teacher, models):
    shape = teacher['predictions']['source_shape']
    h, w = shape
    pool = []
    for model in models:
        if model['source_sha256'] != teacher['source_sha256'] or model['predictions']['source_shape'] != shape:
            raise ValueError('median proposal source mismatch')
        for row in model['predictions']['merged_predictions']:
            box = list(map(float, row['box_xyxy']))
            if not all(math.isfinite(v) for v in box):
                raise ValueError('nonfinite geometry')
            l, t, r, b = box
            if row['confidence'] > .05 and 16 <= l < r <= w - 16 and 16 <= t < b <= h - 16:
                pool.append((model['weight_sha256'], row))
    pool.sort(key=lambda pair: (-pair[1]['confidence'], pair[1]['class_id'], *pair[1]['box_xyxy'], pair[0]))
    selected = []
    for digest, seed in pool:
        per_weight = {}
        for weight, row in pool:
            if row['class_id'] == seed['class_id'] and box_iou(row['box_xyxy'], seed['box_xyxy']) >= .5:
                per_weight.setdefault(weight, row)
        if len(per_weight) < 2:
            continue
        box = [float(statistics.median([row['box_xyxy'][i] for row in per_weight.values()])) for i in range(4)]
        l, t, r, b = box
        if not (16 <= l < r <= w - 16 and 16 <= t < b <= h - 16):
            continue
        supporters = {weight: row for weight, row in per_weight.items() if box_iou(row['box_xyxy'], box) >= .5}
        if len(supporters) < 2 or any(box_iou(box, row['box_xyxy']) >= .5 for row in selected):
            continue
        candidate = copy.deepcopy(seed)
        candidate.update(box_xyxy=box, semantic_detector_weight_sha256=digest,
            semantic_model_vote_sha256=sorted(supporters), semantic_proposal_floor=.05,
            median_geometry=True, median_seed_box=copy.deepcopy(seed['box_xyxy']),
            median_support_boxes={weight: copy.deepcopy(row['box_xyxy']) for weight, row in supporters.items()},
            median_unique_checkpoint_count=len(supporters))
        selected.append(candidate)
    return selected
