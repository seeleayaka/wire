"""Class certainty gate, localization-agreement rank; no GT at inference."""
import math
import statistics
from novel_box_geometry import is_duplicate


def validate_geometry_score(row):
    values=row.get('localization_voter_best_IoU',{})
    votes=row.get('semantic_model_vote_sha256',[])
    if set(values)!=set(votes) or len(values)<3:raise ValueError('Geometry voter correspondence missing')
    if not all(math.isfinite(float(v)) and .5<=v<=1 for v in values.values()):raise ValueError('Invalid voter geometry score')
    return float(statistics.median(values.values()))


def select(current,candidates,probabilities,head_sha):
    from inspection_agent.paired_native_pose_features import select as validate_original
    from inspection_agent.paired_port_features import select as append_original
    # Validate all original probability contracts, including suppressed rows.
    validate_original(current,candidates,probabilities,head_sha)
    ranked=[]
    for row,prob in zip(candidates,probabilities):
        cls=max(range(3),key=lambda i:prob[i])
        if cls!=row['class_id']+1 or prob[cls]<.98 or len(set(row['semantic_model_vote_sha256']))<3:continue
        geometry=validate_geometry_score(row)
        if is_duplicate(row['box_xyxy'],current['all_predictions']):continue
        ranked.append((row,prob,geometry))
    excluded=set()
    for _ in range(len(ranked)+1):
        chosen={}
        for row,prob,geometry in ranked:
            key=(row['class_id'],*row['box_xyxy'])
            if key in excluded:continue
            parent=row['pose_parent_seed_id'];rank=(-geometry,-prob[row['class_id']+1],*row['box_xyxy'],row['class_id'])
            if parent not in chosen or rank<chosen[parent][0]:chosen[parent]=(rank,row,prob)
        rows=[v[1] for v in chosen.values()];scores=[v[2] for v in chosen.values()]
        out=append_original(current,rows,scores,head_sha);kept=[];bad=None
        for row in out['paired_semantic_additions']:
            if is_duplicate(row['box_xyxy'],kept):bad=row;break
            kept.append(row)
        if bad is None:return out
        excluded.add((bad['class_id'],*bad['box_xyxy']))
    raise AssertionError('Finite consensus duplicate exclusion failed')
