"""Conservative perturbation consensus, without boosting raw probabilities."""
import copy
from paired_pose_search import select as pose_select
from paired_shape_features import crop_signature
from inspection_agent.port_tiling import box_iou


def select(current, candidates, probabilities, head_sha):
    # Validate every probability before constructing the additional veto.
    pose_select(current, candidates, probabilities, head_sha)
    accepted = []
    for row, prob in zip(candidates, probabilities):
        cls = row['class_id'] + 1
        if max(range(3), key=lambda i: prob[i]) != cls or prob[cls] < .98: continue
        signature = tuple(crop_signature(row['box_xyxy'], s) for s in (1.5, 3.))
        partners = [other for other, score in zip(candidates, probabilities)
            if other['pose_parent_seed_id'] == row['pose_parent_seed_id']
            and other['class_id'] == row['class_id']
            and tuple(crop_signature(other['box_xyxy'], s) for s in (1.5, 3.)) != signature
            and box_iou(other['box_xyxy'], row['box_xyxy']) >= .5
            and max(range(3), key=lambda i: score[i]) == cls and score[cls] >= .98]
        if partners: accepted.append((row, prob))
    return pose_select(current, [copy.deepcopy(p) for p,v in accepted], [v for p,v in accepted], head_sha)
