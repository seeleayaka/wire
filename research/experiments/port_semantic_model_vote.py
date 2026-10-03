"""New proposal channel: two distinct checkpoint votes plus frozen semantics."""
import copy
from core_port_precision_policy import iou
from port_semantic_support_policy import complete

PROPOSAL_FLOOR=.05
def proposals(teacher,models):
    shape=teacher['predictions']['source_shape'];pool=[]
    for model in models:
        if model['source_sha256']!=teacher['source_sha256'] or model['predictions']['source_shape']!=shape:
            raise ValueError('semantic_model_vote_source_mismatch')
        for row in model['predictions']['merged_predictions']:
            if row['confidence']>PROPOSAL_FLOOR and complete(row,shape):
                pool.append((model['weight_sha256'],row))
    selected=[]
    for digest,row in sorted(pool,key=lambda pair:-pair[1]['confidence']):
        votes={weight for weight,other in pool if other['class_id']==row['class_id'] and
            iou(other['box_xyxy'],row['box_xyxy'])>=.5}
        if len(votes)<2:continue
        if any(iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in selected):continue
        candidate=copy.deepcopy(row);candidate.update(semantic_detector_weight_sha256=digest,
            semantic_model_vote_sha256=sorted(votes),semantic_proposal_floor=PROPOSAL_FLOOR)
        selected.append(candidate)
    return selected
