"""Research affine fallback with identical static-match and held-out gates.

Fewer geometric degrees of freedom, not a lower acceptance threshold. Neither
background registration nor affine support verifies object identity/continuity.
"""
import hashlib
import math
import cv2
import numpy as np
from heldout_anchor_pose import POLICY
from local_anchor_pose import project_points
from prepare_mendeley_scope import inspection_scope

def fit_affine(source,destination,anchor,context,global_matrix):
    src=np.asarray(source,np.float32);dst=np.asarray(destination,np.float32)
    if src.shape!=dst.shape or src.ndim!=2 or src.shape[1]!=2 or not np.isfinite(src).all() or not np.isfinite(dst).all():raise ValueError('finite paired correspondences required')
    result=dict(id=anchor['id'],localization_proposal_supported=False,identity_verified=False,
        plug_seating_assessed=False,electrical_continuity='not_assessed',model='affine_6dof',
        policy=dict(POLICY),no_acceptance_threshold_change=True)
    check=np.array([int.from_bytes(hashlib.sha256(f'{round(float(x))},{round(float(y))}'.encode('ascii')).digest()[:4],'big')%3==0 for x,y in dst],bool)
    train=~check;result.update(training_count=int(train.sum()),heldout_count=int(check.sum()),
        heldout_points_used_to_fit=False,split_before_fit=True)
    if train.sum()<12 or check.sum()<8:
        result['reason']='too_few_disjoint_fit_and_check_matches';return result
    affine,inliers=cv2.estimateAffine2D(src[train],dst[train],method=cv2.RANSAC,
        ransacReprojThreshold=4,maxIters=10000,confidence=.995,refineIters=10)
    if affine is None or inliers is None:
        result['reason']='no_valid_training_transform';return result
    matrix=np.vstack([affine,[0,0,1]])
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)<3:
        result['reason']='no_valid_training_transform';return result
    active=inliers.ravel().astype(bool);count=int(active.sum());td=dst[train]
    training_error=np.linalg.norm(project_points(src[train],matrix)-td,axis=1)
    heldout_error=np.linalg.norm(project_points(src[check],matrix)-dst[check],axis=1)
    left,top,right,bottom=context
    cells={(min(1,int((x-left)*2/(right-left))),min(1,int((y-top)*2/(bottom-top)))) for x,y in td[active]}
    l,t,r,b=anchor['bbox_xyxy'];corners=np.array([[l,t],[r,t],[r,b],[l,b]],np.float32)
    hull=cv2.convexHull(td[active]) if count>=3 else None
    inside=bool(hull is not None and all(cv2.pointPolygonTest(hull,tuple(map(float,p)),False)>=0 for p in corners))
    median=float(np.median(training_error[active])) if count else float('inf')
    check_median=float(np.median(heldout_error));ratio=float((heldout_error<=4).mean())
    gates=dict(training_inliers=count>=12,training_inlier_ratio=bool(count/train.sum()>=.5),
        training_spatial_cells=len(cells)>=3,training_reprojection=median<=2.5,
        heldout_count=bool(check.sum()>=8),heldout_reprojection=check_median<=2.5,
        heldout_support_ratio=ratio>=.5,anchor_in_training_hull=inside)
    local=project_points(corners,np.linalg.inv(matrix));global_=project_points(corners,np.linalg.inv(global_matrix))
    result.update(gates=gates,localization_proposal_supported=all(gates.values()),
        inspection_to_reference_local=matrix.tolist(),inspection_anchor_polygon_xy=local.tolist(),
        training_inliers=count,training_median_error_px=median,heldout_median_error_px=check_median,
        heldout_support_ratio=ratio,local_global_corner_disagreement_px=float(np.linalg.norm(local-global_,axis=1).max()),
        reason='affine_heldout_spatial_support_not_identity' if all(gates.values()) else 'affine_heldout_or_spatial_support_failed')
    return result

def localize_affine(reference,inspection,anchor,global_matrix):
    b=anchor['bbox_xyxy'];margin=3*max(b[2]-b[0],b[3]-b[1]);h,w=reference.shape[:2]
    box=[math.floor(b[0]-margin),math.floor(b[1]-margin),math.ceil(b[2]+margin),math.ceil(b[3]+margin)]
    if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):return dict(id=anchor['id'],localization_proposal_supported=False,reason='reference_context_outside_image')
    ins_box=inspection_scope(box,np.asarray(global_matrix),[inspection.shape[1],inspection.shape[0]])
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.014,edgeThreshold=12)
    kr,dr=sift.detectAndCompute(cv2.cvtColor(reference[box[1]:box[3],box[0]:box[2]],cv2.COLOR_RGB2GRAY),None)
    ki,di=sift.detectAndCompute(cv2.cvtColor(inspection[ins_box[1]:ins_box[3],ins_box[0]:ins_box[2]],cv2.COLOR_RGB2GRAY),None)
    if dr is None or di is None:return dict(id=anchor['id'],localization_proposal_supported=False,reason='no_static_texture')
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def ratio(rows):return {p[0].queryIdx:p[0].trainIdx for p in rows if len(p)==2 and p[0].distance<.70*p[1].distance}
    forward=ratio(matcher.knnMatch(di,dr,k=2));backward=ratio(matcher.knnMatch(dr,di,k=2));src=[];dst=[]
    for i,r in sorted(forward.items()):
        if backward.get(r)!=i:continue
        ref=np.array(kr[r].pt)+box[:2];ins=np.array(ki[i].pt)+ins_box[:2];mapped=project_points([ins],global_matrix)[0]
        if any(b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3] for p in [ref,mapped]):continue
        src.append(ins);dst.append(ref)
    result=fit_affine(np.array(src).reshape(-1,2),np.array(dst).reshape(-1,2),anchor,box,global_matrix)
    result.update(reference_context_xyxy=box,inspection_context_xyxy=ins_box,
        mutual_matches=len(src),reference_anchor_interior_excluded=True)
    return result
