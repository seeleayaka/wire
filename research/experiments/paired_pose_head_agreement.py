"""Two classifiers must accept the SAME native pose; scores are not boosted."""
import copy
from paired_pose_search import select as pose_select


def select(current, candidates, original_scores, auxiliary_scores, original_sha, auxiliary_sha):
    if len(candidates) != len(auxiliary_scores): raise ValueError('auxiliary count mismatch')
    pose_select(current, candidates, original_scores, original_sha)
    pose_select(current, candidates, auxiliary_scores, auxiliary_sha)
    rows = []; scores = []
    for row, old, new in zip(candidates, original_scores, auxiliary_scores):
        cls = row['class_id'] + 1
        if any(max(range(3), key=lambda i:p[i]) != cls or p[cls] < .98 for p in (old,new)): continue
        native = copy.deepcopy(row)
        native.update(pose_auxiliary_head_sha256=auxiliary_sha, pose_auxiliary_probability=float(new[cls]),
            pose_classifier_agreement=True)
        rows.append(native); scores.append(old)
    return pose_select(current, rows, scores, original_sha)
