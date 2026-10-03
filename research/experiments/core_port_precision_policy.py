"""Fixed precision challengers: no labels, scene coordinates or image names."""
import copy
from core_port_budget_policy import bounded_expand


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union > 0 else 0.


def select(prediction, mode):
    rows = bounded_expand(prediction['merged_predictions'])
    if mode == 'bounded':
        return rows
    if mode == 'high_score':
        return [p for p in rows if p['confidence'] > .5]
    if mode == 'high_score_complete_frame':
        height, width = prediction['source_shape']
        # Same 16px margin already used at artificial tile cuts; all four edges.
        return [p for p in rows if p['confidence'] > .5 and
                p['box_xyxy'][0] > 16 and p['box_xyxy'][1] > 16 and
                p['box_xyxy'][2] < width-16 and p['box_xyxy'][3] < height-16]
    if mode != 'weak_overlap_consensus':
        raise ValueError('unknown precision policy')
    height, width = prediction['source_shape']
    result = []
    for p in rows:
        if p['confidence'] > .5:
            result.append(p)
            continue
        l, t, r, b = p['box_xyxy']
        opportunities = set()
        for index, (x, y, right, bottom) in enumerate(prediction['windows']):
            if (l > x + (16 if x else -1) and t > y + (16 if y else -1)
                    and r < right - (16 if right < width else -1)
                    and b < bottom - (16 if bottom < height else -1)):
                opportunities.add(index)
        support = {q['source_tile'] for q in prediction['edge_kept_predictions']
                   if q['confidence'] > .25 and q['class_id'] == p['class_id']
                   and iou(q['box_xyxy'], p['box_xyxy']) >= .5}
        if len(opportunities) < 2 or len(support & opportunities) >= 2:
            result.append(p)
    return copy.deepcopy(result)
