"""Reference-defined local static-context registration; never cable correctness.

The anchor interior is excluded from feature matching. Background localization
may establish a ROI proposal, not plug seating, cable identity or continuity.
"""
import math

import cv2
import numpy as np

from prepare_mendeley_scope import inspection_scope


def project_points(points, matrix):
    q=np.column_stack((np.asarray(points,float),np.ones(len(points)))) @ np.asarray(matrix,float).T
    if not np.isfinite(q).all() or np.any(np.abs(q[:,2])<1e-9):raise ValueError('invalid projective points')
    return q[:,:2]/q[:,2:]


def localize_anchor(reference,inspection,anchor,inspection_to_reference):
    b=anchor['bbox_xyxy'];width,height=reference.shape[1],reference.shape[0]
    margin=3*max(b[2]-b[0],b[3]-b[1])
    ref_box=[math.floor(b[0]-margin),math.floor(b[1]-margin),math.ceil(b[2]+margin),math.ceil(b[3]+margin)]
    result={'id':anchor['id'],'anchor_kind':anchor['kind'],'identity_state':'unverified',
        'reliable_localization':False,'electrical_continuity':'not_assessed',
        'plug_seating_assessed':False,'reference_box_xyxy':list(b),
        'context_margin_anchor_maxside_ratio':3,'confirmed':False}
    if not (0<=ref_box[0]<ref_box[2]<=width and 0<=ref_box[1]<ref_box[3]<=height):
        result['reason']='reference_context_outside_image';return result
    ins_box=inspection_scope(ref_box,np.asarray(inspection_to_reference,float),[inspection.shape[1],inspection.shape[0]])
    ref_crop=reference[ref_box[1]:ref_box[3],ref_box[0]:ref_box[2]]
    ins_crop=inspection[ins_box[1]:ins_box[3],ins_box[0]:ins_box[2]]
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.014,edgeThreshold=12)
    kr,dr=sift.detectAndCompute(cv2.cvtColor(ref_crop,cv2.COLOR_RGB2GRAY),None)
    ki,di=sift.detectAndCompute(cv2.cvtColor(ins_crop,cv2.COLOR_RGB2GRAY),None)
    result.update(reference_context_xyxy=ref_box,inspection_context_xyxy=ins_box,
                  reference_keypoints=len(kr),inspection_keypoints=len(ki))
    if dr is None or di is None:
        result['reason']='insufficient_static_context_texture';return result
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def ratio(rows):return {p[0].queryIdx:p[0].trainIdx for p in rows if len(p)==2 and p[0].distance<.70*p[1].distance}
    forward=ratio(matcher.knnMatch(di,dr,k=2));backward=ratio(matcher.knnMatch(dr,di,k=2))
    source=[];destination=[]
    for i,r in forward.items():
        if backward.get(r)!=i:continue
        ref_point=np.array(kr[r].pt)+ref_box[:2];ins_point=np.array(ki[i].pt)+ins_box[:2]
        mapped=project_points([ins_point],inspection_to_reference)[0]
        # Do not use the moving plug/lead interior to certify its own position.
        if any(b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3] for p in [ref_point,mapped]):continue
        source.append(ins_point);destination.append(ref_point)
    result['mutual_static_context_matches']=len(source)
    if len(source)<12:
        result['reason']='too_few_mutual_static_matches';return result
    source,destination=np.array(source,np.float32),np.array(destination,np.float32)
    matrix,mask=cv2.findHomography(source,destination,method=getattr(cv2,'USAC_MAGSAC',cv2.RANSAC),
        ransacReprojThreshold=4,maxIters=10000,confidence=.995)
    if matrix is None or mask is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)<3:
        result['reason']='no_valid_local_transform';return result
    active=mask.ravel().astype(bool);count=int(active.sum())
    errors=np.linalg.norm(project_points(source,matrix)-destination,axis=1)
    cells={(min(1,int((p[0]-ref_box[0])*2/(ref_box[2]-ref_box[0]))),
            min(1,int((p[1]-ref_box[1])*2/(ref_box[3]-ref_box[1])))) for p in destination[active]}
    median=float(np.median(errors[active])) if count else float('inf')
    corners=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]
    local=project_points(corners,np.linalg.inv(matrix));global_=project_points(corners,np.linalg.inv(inspection_to_reference))
    disagreement=float(np.max(np.linalg.norm(local-global_,axis=1)))
    # Fixed diagnostic gates, not a changed production registration/acceptance gate.
    gates={'inliers':count>=12,'inlier_ratio':count/len(source)>=.5,'spatial_cells':len(cells)>=3,
           'reprojection':median<=2.5,'local_global_agreement':disagreement<=max(2.5,.10*min(b[2]-b[0],b[3]-b[1]))}
    result.update(inliers=count,inlier_ratio=count/len(source),inlier_cells=len(cells),
        median_error_px=median,local_global_corner_disagreement_px=disagreement,gates=gates,
        inspection_to_reference_local=matrix.tolist(),inspection_anchor_polygon_xy=local.tolist(),
        reliable_localization=all(gates.values()),anchor_interior_feature_matches_excluded=True,
        identity_state='localized_reference_anchor_proposal' if all(gates.values()) else 'unverified',
        reason='static_context_localization_supported_not_connection_proof' if all(gates.values()) else 'local_pose_gate_failed')
    return result
