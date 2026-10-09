"""L1/square-root SIFT descriptor experiment; original heldout geometry unchanged."""
import math
import cv2
import numpy as np
from heldout_anchor_pose import fit_with_holdout
from local_anchor_pose import project_points
from prepare_mendeley_scope import inspection_scope


def normalize(descriptors):
    values=np.asarray(descriptors)
    if values.ndim!=2 or values.shape[1]!=128 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('finite nonnegative128 SIFT dimensions required')
    mass=values.sum(1,keepdims=True)
    if np.any(mass<=0): raise ValueError('zero descriptor cannot support localization')
    return np.sqrt(values/mass).astype(np.float32)


def localize_root(reference,inspection,anchor,global_matrix):
    b=anchor['bbox_xyxy']; margin=3*max(b[2]-b[0],b[3]-b[1]); h,w=reference.shape[:2]
    box=[math.floor(b[0]-margin),math.floor(b[1]-margin),math.ceil(b[2]+margin),math.ceil(b[3]+margin)]
    if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
        return dict(id=anchor['id'],localization_proposal_supported=False,reason='reference_context_outside_image')
    ins_box=inspection_scope(box,np.asarray(global_matrix),[inspection.shape[1],inspection.shape[0]])
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.014,edgeThreshold=12)
    kr,dr=sift.detectAndCompute(cv2.cvtColor(reference[box[1]:box[3],box[0]:box[2]],cv2.COLOR_RGB2GRAY),None)
    ki,di=sift.detectAndCompute(cv2.cvtColor(inspection[ins_box[1]:ins_box[3],ins_box[0]:ins_box[2]],cv2.COLOR_RGB2GRAY),None)
    if dr is None or di is None:
        return dict(id=anchor['id'],localization_proposal_supported=False,reason='no_static_texture')
    dr,di=normalize(dr),normalize(di)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def ratio(rows):
        return {r[0].queryIdx:r[0].trainIdx for r in rows if len(r)==2 and r[0].distance<.70*r[1].distance}
    forward=ratio(matcher.knnMatch(di,dr,k=2)); backward=ratio(matcher.knnMatch(dr,di,k=2))
    source=[]; destination=[]
    for i,r in sorted(forward.items()):
        if backward.get(r)!=i: continue
        ref=np.array(kr[r].pt)+box[:2]; ins=np.array(ki[i].pt)+ins_box[:2]
        mapped=project_points([ins],global_matrix)[0]
        if any(b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3] for p in [ref,mapped]): continue
        source.append(ins); destination.append(ref)
    result=fit_with_holdout(np.asarray(source).reshape(-1,2),np.asarray(destination).reshape(-1,2),
                            anchor,box,global_matrix)
    result.update(reference_context_xyxy=box,inspection_context_xyxy=ins_box,
                  reference_anchor_interior_excluded=True,descriptor_transform='L1_then_square_root',
                  reference_keypoints=len(kr),inspection_keypoints=len(ki),
                  source_correspondences_xy=np.asarray(source).reshape(-1,2).tolist(),
                  reference_correspondences_xy=np.asarray(destination).reshape(-1,2).tolist())
    return result
