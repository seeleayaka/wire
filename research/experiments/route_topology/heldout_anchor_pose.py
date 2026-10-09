"""Separate research localization using held-out static image correspondences.

Global-local disagreement remains visible; the old agreement gate is not changed.
This measures spatial localization only, NOT port identity or plug seating.
"""
import hashlib
import math

import cv2
import numpy as np

from local_anchor_pose import project_points
from prepare_mendeley_scope import inspection_scope

POLICY={'context_margin_ratio':3,'mutual_ratio':.70,'training_inliers_min':12,
    'training_inlier_ratio_min':.5,'training_cells_min':3,'heldout_count_min':8,
    'heldout_support_ratio_min':.5,'heldout_median_error_max_px':2.5,
    'heldout_point_support_max_px':4,'training_reprojection_max_px':2.5,
    'holdout_hash_modulus':3,'require_anchor_corners_in_training_hull':True}


def fit_with_holdout(source,destination,anchor,reference_context,global_matrix):
    source=np.asarray(source,np.float32);destination=np.asarray(destination,np.float32)
    result={'id':anchor['id'],'localization_proposal_supported':False,
        'identity_verified':False,'plug_seating_assessed':False,'electrical_continuity':'not_assessed',
        'global_agreement_gate_replaced':False,'policy':dict(POLICY)}
    if (source.shape!=destination.shape or source.ndim!=2 or source.shape[1]!=2
            or not np.isfinite(source).all() or not np.isfinite(destination).all()):
        raise ValueError('finite paired source-pixel correspondences required')
    # Split before fitting or outlier selection. Duplicate descriptor matches at
    # the same rounded reference location always go to the same side.
    heldout=np.array([int.from_bytes(hashlib.sha256(
        f'{round(float(x))},{round(float(y))}'.encode('ascii')).digest()[:4],'big')%3==0
        for x,y in destination],bool)
    training=~heldout;result.update(mutual_matches=len(source),training_count=int(training.sum()),
        heldout_count=int(heldout.sum()),split_before_fit=True,heldout_points_used_to_fit=False)
    if training.sum()<12 or heldout.sum()<8:
        result['reason']='too_few_disjoint_fit_and_check_matches';return result
    matrix,mask=cv2.findHomography(source[training],destination[training],
        method=getattr(cv2,'USAC_MAGSAC',cv2.RANSAC),ransacReprojThreshold=4,maxIters=10000,confidence=.995)
    if matrix is None or mask is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)<3:
        result['reason']='no_valid_training_transform';return result
    active=mask.ravel().astype(bool);count=int(active.sum())
    train_destination=destination[training];train_error=np.linalg.norm(
        project_points(source[training],matrix)-train_destination,axis=1)
    check_error=np.linalg.norm(project_points(source[heldout],matrix)-destination[heldout],axis=1)
    bx,by,cx,cy=reference_context
    cells={(min(1,int((x-bx)*2/(cx-bx))),min(1,int((y-by)*2/(cy-by))))
        for x,y in train_destination[active]}
    a,b,c,d=anchor['bbox_xyxy'];corners=np.array([[a,b],[c,b],[c,d],[a,d]],np.float32)
    hull=cv2.convexHull(train_destination[active]) if count>=3 else None
    inside=bool(hull is not None and all(cv2.pointPolygonTest(hull,tuple(map(float,p)),False)>=0 for p in corners))
    train_median=float(np.median(train_error[active])) if count else float('inf')
    check_median=float(np.median(check_error));check_ratio=float((check_error<=4).mean())
    gates={'training_inliers':count>=12,'training_inlier_ratio':bool(count/training.sum()>=.5),
        'training_spatial_cells':len(cells)>=3,'training_reprojection':train_median<=2.5,
        'heldout_count':bool(heldout.sum()>=8),'heldout_reprojection':check_median<=2.5,
        'heldout_support_ratio':check_ratio>=.5,'anchor_in_training_hull':inside}
    local=project_points(corners,np.linalg.inv(matrix));global_=project_points(corners,np.linalg.inv(global_matrix))
    result.update(training_inliers=count,training_inlier_ratio=count/int(training.sum()),
        training_cells=len(cells),training_median_error_px=train_median,heldout_median_error_px=check_median,
        heldout_support_ratio=check_ratio,gates=gates,inspection_to_reference_local=matrix.tolist(),
        inspection_anchor_polygon_xy=local.tolist(),
        local_global_corner_disagreement_px=float(np.max(np.linalg.norm(local-global_,axis=1))),
        localization_proposal_supported=all(gates.values()),
        reason='heldout_spatial_localization_supported_not_semantic_identity' if all(gates.values())
            else 'heldout_or_spatial_support_failed')
    return result


def localize(reference,inspection,anchor,global_matrix):
    b=anchor['bbox_xyxy'];margin=3*max(b[2]-b[0],b[3]-b[1]);w,h=reference.shape[1],reference.shape[0]
    box=[math.floor(b[0]-margin),math.floor(b[1]-margin),math.ceil(b[2]+margin),math.ceil(b[3]+margin)]
    if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
        return {'id':anchor['id'],'localization_proposal_supported':False,'reason':'reference_context_outside_image'}
    ins_box=inspection_scope(box,np.asarray(global_matrix,float),[inspection.shape[1],inspection.shape[0]])
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.014,edgeThreshold=12)
    kr,dr=sift.detectAndCompute(cv2.cvtColor(reference[box[1]:box[3],box[0]:box[2]],cv2.COLOR_RGB2GRAY),None)
    ki,di=sift.detectAndCompute(cv2.cvtColor(inspection[ins_box[1]:ins_box[3],ins_box[0]:ins_box[2]],cv2.COLOR_RGB2GRAY),None)
    if dr is None or di is None:
        return {'id':anchor['id'],'localization_proposal_supported':False,'reason':'no_static_texture'}
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def ratio(rows):return {p[0].queryIdx:p[0].trainIdx for p in rows if len(p)==2 and p[0].distance<.70*p[1].distance}
    forward=ratio(matcher.knnMatch(di,dr,k=2));backward=ratio(matcher.knnMatch(dr,di,k=2))
    src=[];dst=[]
    for i,r in sorted(forward.items()):
        if backward.get(r)!=i:continue
        ref=np.array(kr[r].pt)+box[:2];ins=np.array(ki[i].pt)+ins_box[:2]
        mapped=project_points([ins],global_matrix)[0]
        if any(b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3] for p in [ref,mapped]):continue
        src.append(ins);dst.append(ref)
    result=fit_with_holdout(np.array(src).reshape(-1,2),np.array(dst).reshape(-1,2),anchor,box,global_matrix)
    result.update(reference_context_xyxy=box,inspection_context_xyxy=ins_box,
        reference_anchor_interior_excluded=True,reference_keypoints=len(kr),inspection_keypoints=len(ki))
    return result
