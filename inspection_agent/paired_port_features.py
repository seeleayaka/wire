"""Frozen paired crop features and additive native selector; no GT at inference."""
import copy
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from inspection_agent.port_tiling import box_iou
from inspection_agent.teacher_student_port_support import valid

CONTEXT_SCALES=(1.5,3.0)
PROBABILITY_GATE=.98
MIN_VALID_COVERAGE=.85


def context_box(box,scale):
    l,t,r,b=map(float,box);cx,cy=(l+r)/2,(t+b)/2
    side=max(24.,max(r-l,b-t)*scale)
    return [cx-side/2,cy-side/2,cx+side/2,cy+side/2]


def crop_tensor(image,box,scale):
    l,t,r,b=map(float,box)
    if not np.isfinite([l,t,r,b]).all() or r<=l or b<=t:raise ValueError('invalid crop box')
    cx,cy=(l+r)/2,(t+b)/2;side=max(24.,max(r-l,b-t)*scale)
    left,top=math.floor(cx-side/2),math.floor(cy-side/2);right,bottom=math.ceil(cx+side/2),math.ceil(cy+side/2)
    h,w=image.shape[:2]
    if left>=w or top>=h or right<=0 or bottom<=0:raise ValueError('crop outside image')
    padded=cv2.copyMakeBorder(image[max(0,top):min(h,bottom),max(0,left):min(w,right)],
        max(0,-top),max(0,bottom-h),max(0,-left),max(0,right-w),cv2.BORDER_CONSTANT,value=(124,116,104))
    rgb=cv2.cvtColor(cv2.resize(padded,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    tensor=torch.from_numpy(rgb).permute(2,0,1).float()/255.
    return (tensor-torch.tensor([.485,.456,.406]).view(3,1,1))/torch.tensor([.229,.224,.225]).view(3,1,1)


def embeddings(model,image,boxes):
    tensors=[crop_tensor(image,box,scale) for box in boxes for scale in CONTEXT_SCALES];vectors=[]
    with torch.inference_mode():
        for start in range(0,len(tensors),8):
            result=model.forward_features(torch.stack(tensors[start:start+8]))
            patches=result['x_norm_patchtokens'].reshape(-1,16,16,384)
            cls=F.normalize(result['x_norm_clstoken'],dim=1)
            center=F.normalize(patches[:,6:10,6:10].mean(dim=(1,2)),dim=1)
            vectors.append(torch.cat((cls,center),dim=1))
    if not vectors:return torch.empty((0,1536))
    return torch.cat(vectors).reshape(len(boxes),1536)/2.


def paired_features(observed,expected):
    if observed.ndim!=2 or observed.shape!=expected.shape or observed.shape[1]!=1536:raise ValueError('paired embedding shape mismatch')
    if not torch.isfinite(observed).all() or not torch.isfinite(expected).all():raise ValueError('nonfinite paired embeddings')
    return torch.cat((observed,expected,(observed-expected).abs(),observed*expected),dim=1)


def expected_in_source(reference,matrix,shape):
    h,w=shape;matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)!=3 or min(h,w)<1:raise ValueError('invalid reference alignment')
    inverse=np.linalg.inv(matrix)
    expected=cv2.warpPerspective(reference,inverse,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=cv2.warpPerspective(np.full(reference.shape[:2],255,np.uint8),inverse,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    valid=cv2.erode(valid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
    return expected,valid


def context_valid_fraction(box,mask,scale):
    l,t,r,b=context_box(box,scale);l,t,r,b=math.floor(l),math.floor(t),math.ceil(r),math.ceil(b)
    h,w=mask.shape;part=mask[max(0,t):min(h,b),max(0,l):min(w,r)] if r>0 and b>0 and l<w and t<h else np.zeros((0,0),bool)
    return float(part.sum())/max(1,(r-l)*(b-t))


def valid_boxes(boxes,mask):
    return [i for i,box in enumerate(boxes) if all(context_valid_fraction(box,mask,s)>=MIN_VALID_COVERAGE for s in CONTEXT_SCALES)]


def proposals(teacher,models):
    shape=teacher['predictions']['source_shape'];pool=[]
    for model in models:
        if model['source_sha256']!=teacher['source_sha256'] or model['predictions']['source_shape']!=shape:raise ValueError('semantic_model_vote_source_mismatch')
        for row in model['predictions']['merged_predictions']:
            l,t,r,b=row['box_xyxy'];h,w=shape
            if row['confidence']>.05 and 16<=l<r<=w-16 and 16<=t<b<=h-16:pool.append((model['weight_sha256'],row))
    selected=[]
    for digest,row in sorted(pool,key=lambda pair:-pair[1]['confidence']):
        votes={weight for weight,other in pool if other['class_id']==row['class_id'] and box_iou(other['box_xyxy'],row['box_xyxy'])>=.5}
        if len(votes)<2:continue
        if any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in selected):continue
        candidate=copy.deepcopy(row);candidate.update(semantic_detector_weight_sha256=digest,semantic_model_vote_sha256=sorted(votes),semantic_proposal_floor=.05)
        selected.append(candidate)
    return selected


def select(current,candidates,probabilities,head_sha):
    if len(candidates)!=len(probabilities):raise ValueError('probability/proposal mismatch')
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    if len(current['primary'])>5 or remaining<0:raise ValueError('invalid current budget')
    choices=[]
    for row,prob in zip(candidates,probabilities):
        if len(prob)!=3 or not all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in prob) or abs(sum(prob)-1)>1e-5:raise ValueError('invalid classifier probability')
        cls=max(range(3),key=lambda c:prob[c])
        if cls==0 or prob[cls]<PROBABILITY_GATE or cls!=row['class_id']+1:continue
        if not valid(row):raise ValueError('invalid native proposal')
        if len(set(row.get('semantic_model_vote_sha256',[])))<2:raise ValueError('distinct proposal votes missing')
        added=copy.deepcopy(row);added.update(proposal_detector_score=row['confidence'],confidence=float(prob[cls]),
            paired_semantic_probability=float(prob[cls]),paired_head_sha256=head_sha,
            evidence_tier='observed_expected_port_semantics_experimental_manual_review',automatic_fault_verdict=False)
        choices.append(added)
    choices.sort(key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id']))
    out=copy.deepcopy(current);out['paired_semantic_additions']=[]
    for row in choices:
        if len(out['paired_semantic_additions'])>=remaining:break
        if any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in out['all_predictions']):continue
        out['paired_semantic_additions'].append(row);out['all_predictions'].append(row)
    assert out['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(out['all_predictions'])<=len(current['primary'])+5
    return out
