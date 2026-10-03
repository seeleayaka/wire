"""GT-free, three-checkpoint median proposals for already selected boxes."""
import copy
import math
import statistics
from inspection_agent.port_tiling import box_iou


def proposals(teacher, models, current):
    h,w = teacher['predictions']['source_shape']; pool = []
    for model in models:
        if model['source_sha256'] != teacher['source_sha256'] or model['predictions']['source_shape'] != [h,w]:
            raise ValueError('existing-geometry source mismatch')
        for row in model['predictions']['merged_predictions']:
            l,t,r,b = row['box_xyxy']
            if row['confidence'] > .05 and all(math.isfinite(float(v)) for v in (l,t,r,b)) and 16 <= l < r <= w-16 and 16 <= t < b <= h-16:
                pool.append((model['weight_sha256'],row))
    pool.sort(key=lambda p:(-p[1]['confidence'],p[0],*p[1]['box_xyxy']))
    result = []
    for index, selected in enumerate(current['all_predictions']):
        supporters = {}
        for weight,row in pool:
            if row['class_id'] == selected['class_id'] and box_iou(row['box_xyxy'],selected['box_xyxy']) >= .5:
                supporters.setdefault(weight,row)
        if len(supporters) < 3: continue
        box = [float(statistics.median([p['box_xyxy'][axis] for p in supporters.values()])) for axis in range(4)]
        l,t,r,b = box
        if not (16 <= l < r <= w-16 and 16 <= t < b <= h-16): continue
        if box_iou(box,selected['box_xyxy']) < .5 or max(abs(a-b) for a,b in zip(box,selected['box_xyxy'])) < 1.: continue
        votes = {weight for weight,row in supporters.items() if box_iou(box,row['box_xyxy']) >= .5}
        if len(votes) < 3: continue
        if any(j != index and box_iou(box,other['box_xyxy']) >= .5 for j,other in enumerate(current['all_predictions'])): continue
        row = copy.deepcopy(selected)
        row.update(box_xyxy=box,existing_prediction_index=index,
            original_selected_box=copy.deepcopy(selected['box_xyxy']),semantic_model_vote_sha256=sorted(votes))
        result.append(row)
    return result


def refine(current, candidates, probabilities, head_sha):
    if len(candidates) != len(probabilities): raise ValueError('refinement count mismatch')
    output = copy.deepcopy(current); changes = []
    for row,prob in zip(candidates,probabilities):
        if len(prob) != 3 or not all(math.isfinite(float(v)) and 0 <= v <= 1 for v in prob) or abs(sum(prob)-1) > 1e-5:
            raise ValueError('invalid refinement probabilities')
        cls = row['class_id'] + 1
        if max(range(3),key=lambda i:prob[i]) != cls or prob[cls] < .98: continue
        index = row['existing_prediction_index']; old = current['all_predictions'][index]
        if old['class_id'] != row['class_id'] or old['box_xyxy'] != row['original_selected_box'] or len(set(row['semantic_model_vote_sha256'])) < 3:
            raise ValueError('invalid existing refinement association')
        if any(j != index and box_iou(row['box_xyxy'],other['box_xyxy']) >= .5 for j,other in enumerate(output['all_predictions'])): continue
        updated = copy.deepcopy(old)
        updated.update(box_xyxy=copy.deepcopy(row['box_xyxy']),original_selected_box=copy.deepcopy(old['box_xyxy']),
            geometry_refinement_probability=float(prob[cls]),geometry_refinement_head_sha256=head_sha,
            geometry_refinement_vote_sha256=row['semantic_model_vote_sha256'],automatic_fault_verdict=False)
        output['all_predictions'][index] = updated
        changes.append(dict(prediction_index=index,old=copy.deepcopy(old),new=copy.deepcopy(updated)))
    assert len(output['all_predictions']) == len(current['all_predictions'])
    output['existing_geometry_changes'] = changes
    # Other historical selection partitions intentionally remain raw originals.
    # This is an independent native geometry experiment, not a GUI hint payload.
    return output
