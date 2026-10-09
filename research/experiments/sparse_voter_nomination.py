"""Sparse evidence may nominate an action; it never produces a final cue."""
import copy
import math
from unresolved_voter_seeds import pool_rows
from inspection_agent.port_tiling import box_iou
from novel_box_geometry import is_duplicate


def nominees(models, current, source_sha, shape):
    pool = pool_rows(models, source_sha, shape)
    if len({m['weight_sha256'] for m in models}) != 3:
        raise ValueError('Three-checkpoint model universe required')
    pool.sort(key=lambda x: (-x[1]['confidence'], x[1]['class_id'], *x[1]['box_xyxy'], x[0]))
    representatives, result = [], []
    for digest, row in pool:
        if any(r['class_id'] == row['class_id'] and box_iou(r['box_xyxy'], row['box_xyxy']) >= .5
               for r in representatives):
            continue
        representatives.append(row)
        if is_duplicate(row['box_xyxy'], current['all_predictions']):
            continue
        votes = {weight for weight, other in pool if other['class_id'] == row['class_id']
                 and box_iou(row['box_xyxy'], other['box_xyxy']) >= .5}
        if len(votes) not in (1, 2):
            continue
        seed = copy.deepcopy(row)
        seed.update(semantic_model_vote_sha256=sorted(votes), nomination_checkpoint_sha256=digest,
                    action_nomination_only=True, automatic_fault_verdict=False)
        result.append(seed)
    return result


def actions(seeds, probabilities, remaining):
    if len(seeds) != len(probabilities) or not isinstance(remaining, int) or not 0 <= remaining <= 5:
        raise ValueError('Nomination row/budget contract mismatch')
    ranked = []
    for seed, prob in zip(seeds, probabilities):
        if len(prob) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in prob) or abs(sum(prob)-1) > 1e-5:
            raise ValueError('Invalid nomination score')
        if not seed.get('action_nomination_only') or seed.get('automatic_fault_verdict') is not False:
            raise ValueError('Seed must be action-only')
        if len(set(seed['semantic_model_vote_sha256'])) not in (1, 2):
            raise ValueError('Not sparse evidence')
        cls = max(range(3), key=lambda i: prob[i])
        if cls != seed['class_id']+1 or prob[cls] < .98:
            continue
        row = copy.deepcopy(seed)
        row['nomination_probability'] = float(prob[cls])
        ranked.append(row)
    ranked.sort(key=lambda p: (-p['nomination_probability'], -p['confidence'], *p['box_xyxy'], p['class_id']))
    chosen = []
    for row in ranked:
        if len(chosen) >= remaining:
            break
        if not is_duplicate(row['box_xyxy'], chosen):
            chosen.append(row)
    return chosen
