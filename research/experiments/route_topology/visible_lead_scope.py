"""Image-bound visible device-lead attachment nominations, not terminal topology.

Device lead emergence is explicitly NOT an electrical terminal. This narrower
entry never calls its relation an electrical connection or continuity verdict.
No fragment bridging or branch pruning; original mask geometry gates retained.
"""
from copy import deepcopy
import math

import numpy as np

from core import reviewed, text
from reference_once import validate_binding


def validate_scope(scope, binding):
    validate_binding(binding)
    if (type(scope.get('schema_version')) is not int or scope['schema_version'] != 1
            or scope.get('kind') != 'reference_once_visible_lead_attachment'
            or scope.get('reference_binding') != binding):
        raise ValueError('scope schema or reference fingerprint/frame mismatch')
    text(scope.get('scope_description'), 'scope description')
    anchors = scope.get('anchors')
    if not isinstance(anchors, list) or len(anchors) != 2:
        raise ValueError('exactly two explicitly typed anchors required')
    ids, kinds = set(), set()
    for anchor in anchors:
        identity = text(anchor.get('id'), 'anchor ID')
        box = anchor.get('bbox_xyxy')
        if (identity in ids or anchor.get('kind') not in ('visible_lead_emergence', 'wire_entry_socket')
                or not isinstance(box, list) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in box)
                or not (0<=box[0]<box[2]<=binding['image_size'][0]
                        and 0<=box[1]<box[3]<=binding['image_size'][1])):
            raise ValueError('invalid anchor identity/type/coordinates')
        ids.add(identity); kinds.add(anchor['kind'])
    if len(kinds) != 2:
        raise ValueError('one visible emergence and one wire-entry socket required')
    a,b = [anchor['bbox_xyxy'] for anchor in anchors]
    if min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1]):
        raise ValueError('anchor rectangles overlap')
    if scope.get('expected_visible_attachment') != sorted(ids):
        raise ValueError('expected visible attachment must match declared anchors')
    if scope.get('electrical_terminal_pair_established') is not False:
        raise ValueError('visible lead emergence must not be called an electrical terminal')
    return reviewed(scope.get('reference_review'))


def nominate_view(scope, reference_binding, records, translation=(0,0), inspection_to_reference=None):
    confirmed = validate_scope(scope, reference_binding)
    matrix = np.eye(3) if inspection_to_reference is None else np.asarray(inspection_to_reference,float)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)!=3:
        raise ValueError('invalid source-to-reference mapping')
    if len(translation)!=2 or any(type(v) is not int for v in translation):
        raise ValueError('verified integer crop translation required')
    ids = [r['record_id'] for r in records]
    if len(ids)!=len(set(ids)):
        raise ValueError('duplicate evidence record ID')
    result = {'reference_confirmed':confirmed,'relation_nominations':[], 'audit':[],
        'decision':'insufficient_evidence','electrical_continuity':'not_assessed',
        'electrical_terminal_pair_established':False,'automatic_fault_verdict':False,
        'physical_new_connections':0,'model_observer_count':1}
    for record in records:
        geometry = record['geometry']
        score = record['score']
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('invalid segmentation score')
        reason = ('unresolved_whole_mask_geometry' if geometry.get('geometry_contract_version')!=2
                  or geometry.get('state')!='simple_visible_path' else
                  'crop_boundary_truncation' if record['boundary_truncated'] else
                  'low_segmentation_score' if score<.75 else None)
        audit = {'record_id':record['record_id'],'reason':reason,'score':score,'anchor_hits':[]}
        if reason:
            result['audit'].append(audit); continue
        path = np.asarray(geometry['path_xy'],float)
        tips = np.asarray(geometry['tips_xy'],float)
        if (path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or tips.shape!=(2,2)
                or not np.isfinite(path).all() or not np.isfinite(tips).all()
                or not (np.array_equal(path[[0,-1]],tips) or np.array_equal(path[[0,-1]],tips[::-1]))):
            raise ValueError('actual finite path endpoints required')
        q = np.column_stack((path+translation,np.ones(len(path)))) @ matrix.T
        if np.any(np.abs(q[:,2])<1e-9) or not (np.all(q[:,2]>0) or np.all(q[:,2]<0)):
            audit['reason']='projective_horizon_in_path'
            result['audit'].append(audit); continue
        mapped = q[:,:2]/q[:,2:]
        w,h = reference_binding['image_size']
        if np.any(mapped<0) or np.any(mapped[:,0]>=w) or np.any(mapped[:,1]>=h):
            audit['reason']='mapped_path_outside_reference'
            result['audit'].append(audit); continue
        for x,y in mapped[[0,-1]]:
            audit['anchor_hits'].append([a['id'] for a in scope['anchors']
                if a['bbox_xyxy'][0]<=x<=a['bbox_xyxy'][2] and a['bbox_xyxy'][1]<=y<=a['bbox_xyxy'][3]])
        hits = audit['anchor_hits']
        if any(len(v)!=1 for v in hits) or hits[0]==hits[1]:
            audit['reason']='endpoints_not_uniquely_attached_to_two_scoped_anchors'
        else:
            result['relation_nominations'].append({'anchors':sorted([hits[0][0],hits[1][0]]),
                'record_id':record['record_id'],'mask_array_sha256':record['mask_array_sha256'],
                'semantic_identity_verified':False,'confirmed':False,
                'reference_frame_path_xy':mapped.tolist()})
        result['audit'].append(audit)
    # Even successful geometry is only a nomination until reference calibration
    # and automatic semantic/local-anchor evidence have actually been verified.
    result['reason'] = ('reference_review_pending' if not confirmed else
                        'automatic_lead_semantics_and_local_anchor_identity_unverified')
    return result
