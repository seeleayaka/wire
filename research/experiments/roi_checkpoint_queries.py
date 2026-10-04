"""GT-free completion of each acquired ROI's checkpoint coverage."""
import copy
from unresolved_voter_seeds import windows


def queries(case, control, chosen):
    if case['image'] != control['image'] or case['source_sha256'] != control['source_sha256']:
        raise ValueError('Control source identity mismatch')
    if chosen != control['chosen_seeds'] or len(chosen) != len(control['ROI_evidence']):
        raise ValueError('Control action identity mismatch')
    shapes = {tuple(m['predictions']['source_shape']) for m in case['models']}
    weights = {m['weight_sha256'] for m in case['models']}
    if len(shapes) != 1 or len(weights) != 3:
        raise ValueError('Need exactly three checkpoints in one coordinate frame')
    if any(m['source_sha256'] != case['source_sha256'] for m in case['models']):
        raise ValueError('Model source identity mismatch')
    result = []
    shape = next(iter(shapes))
    for seed, old in zip(chosen, control['ROI_evidence']):
        missing = weights - set(seed['semantic_model_vote_sha256'])
        ws = windows(seed, shape)
        if len(missing) != 1 or old['seed'] != seed or old['windows'] != ws:
            raise ValueError('Control seed/window mismatch')
        if old['missing_weight_sha256'] != next(iter(missing)) or len(old['views']) != 2:
            raise ValueError('Control checkpoint/views mismatch')
        for weight in sorted(weights):
            result.append(dict(seed=copy.deepcopy(seed), weight_sha256=weight,
                               windows=copy.deepcopy(ws),
                               reused_views=copy.deepcopy(old['views']) if weight in missing else None))
    return result
