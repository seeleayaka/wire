"""Attach localization agreement to unchanged source-coordinate proposals."""
from inspection_agent.port_tiling import box_iou


def attach(candidates,models,source_sha,shape):
    import copy
    h,w=shape;pool=[]
    for model in models:
        if model['source_sha256']!=source_sha or model['predictions']['source_shape']!=list(shape):raise ValueError('Fine voter source/frame mismatch')
        for row in model['predictions']['merged_predictions']:
            l,t,r,b=row['box_xyxy']
            if row['confidence']>.05 and 16<=l<r<=w-16 and 16<=t<b<=h-16:pool.append((model['weight_sha256'],row))
    result=copy.deepcopy(candidates)
    for row in result:
        best={}
        for digest,other in pool:
            if other['class_id']!=row['class_id']:continue
            value=box_iou(row['box_xyxy'],other['box_xyxy'])
            if value>=.5:best[digest]=max(value,best.get(digest,0.))
        if sorted(best)!=row['semantic_model_vote_sha256']:raise ValueError('Proposal votes differ from actual model evidence')
        row['localization_voter_best_IoU']=best
    return result
