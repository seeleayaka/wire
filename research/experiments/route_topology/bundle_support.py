"""Raw-pixel bundle support diagnostics, never a single-wire topology verdict.

All components and branches remain in the evidence. A component touching two
scoped anchors is a retrieval diagnostic, not proof of physical continuity or
bundle identity. No dilation, closing, pruning, nearest-point snap or GT lookup.
"""
import hashlib

import cv2
import numpy as np

from core import extract_mask
from visible_lead_scope import validate_scope


def inspect_support(raw,score,record_id,scope,binding,translation=(0,0),matrix=None):
    validate_scope(scope,binding)
    record=extract_mask(raw,score,record_id)
    if len(translation)!=2 or any(type(v) is not int for v in translation):
        raise ValueError('exact integer crop translation required')
    h=np.eye(3) if matrix is None else np.asarray(matrix,float)
    if h.shape!=(3,3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h)!=3:
        raise ValueError('invalid mapping')
    active=(np.asarray(raw)>0).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(active,connectivity=8)
    components=[]
    for component in range(1,count):
        ys,xs=np.where(labels==component)
        q=np.column_stack((xs+translation[0],ys+translation[1],np.ones(len(xs))))@h.T
        if np.any(np.abs(q[:,2])<1e-9) or not (np.all(q[:,2]>0) or np.all(q[:,2]<0)):
            raise ValueError('projective horizon through component')
        projected=q[:,:2]/q[:,2:]
        width,height=binding['image_size']
        if np.any(projected<0) or np.any(projected[:,0]>=width) or np.any(projected[:,1]>=height):
            raise ValueError('mapped component outside bound reference')
        hits={}
        for anchor in scope['anchors']:
            a,b,c,d=anchor['bbox_xyxy']
            hits[anchor['id']]=int(((projected[:,0]>=a)&(projected[:,0]<=c)&
                                  (projected[:,1]>=b)&(projected[:,1]<=d)).sum())
        components.append({'component_index':component,'pixel_count':int(stats[component,cv2.CC_STAT_AREA]),
            'anchor_pixel_support':hits,'both_anchor_support':all(v>0 for v in hits.values())})
    return {'record_id':record_id,'score':score,'raw_mask_array_sha256':record['mask_array_sha256'],
        'whole_mask_geometry':record['geometry'],'boundary_truncated':record['boundary_truncated'],
        'raw_foreground_component_count':count-1,'components':components,
        'component_has_both_anchor_support':any(c['both_anchor_support'] for c in components),
        'all_components_retained':True,'mask_pixels_changed':False,
        'qualified_single_wire_path':record['geometry']['state']=='simple_visible_path'
            and not record['boundary_truncated'] and score>=.75,
        'bundle_semantics_verified':False,'attachment_confirmed':False,
        'electrical_continuity':'not_assessed','decision':'insufficient_evidence',
        'physical_new_connections':0,'model_observer_count':1}
