"""Explicit BUNDLE-level observation, not single-wire or electrical topology.

Multiwire bundles may branch. Every original mask and component stays in the
audit. Only one native component spanning two locally supported anchors can
support a visible bundle observation. No fragment bridging, pruning or snap.
"""
import math
import cv2
import numpy as np
from core import extract_mask
from reference_once import validate_binding
from visible_lead_scope import validate_scope

POLICY={'kind':'scoped_visible_whole_harness','segmentation_score_min':.75,
    'recipe':'wire_harness_plus_reference_anatomy_box','native_connectivity':8,
    'require_unique_native_two_anchor_component':True,'all_original_components_retained':True,
    'branches_allowed_for_bundle_not_single_wire':True,'physical_new_connections':0,
    'model_observer_count':1,'electrical_continuity':'not_assessed'}


def observe_harness(scope,source_binding,poses,masks,socket_state,translation=(0,0),sam_inventory_verified=False):
    validate_scope(scope,scope['reference_binding']);validate_binding(source_binding)
    if socket_state not in ['mating_body_visible','socket_contacts_exposed','uncertain']:
        raise ValueError('typed visible socket state required')
    if len(translation)!=2 or any(type(v) is not int or v<0 for v in translation):
        raise ValueError('exact nonnegative original-crop translation required')
    declared={a['id']:a for a in scope['anchors']};pose_map={p['id']:p for p in poses}
    if len(pose_map)!=len(poses) or not set(pose_map).issubset(declared):raise ValueError('duplicate/unknown anchor pose')
    supported={}
    for identity,p in pose_map.items():
        gates=p.get('gates',{})
        supported[identity]=(p.get('localization_proposal_supported') is True and bool(gates)
            and all(type(v) is bool and v for v in gates.values()))
        if supported[identity]:
            h=np.asarray(p.get('inspection_to_reference_local'),float)
            if h.shape!=(3,3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h)!=3:
                raise ValueError('invalid local anchor mapping')
    ids=set();audit=[];eligible=[];high_socket_touch=False
    socket_id=next(a['id'] for a in scope['anchors'] if a['kind']=='wire_entry_socket')
    for item in masks:
        identity=item['record_id'];score=item['score']
        if identity in ids:raise ValueError('duplicate native mask record')
        ids.add(identity)
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('invalid SAM score')
        raw=np.asarray(item['raw'])
        if raw.ndim!=2 or raw.shape[0]+translation[1]>source_binding['image_size'][1] or raw.shape[1]+translation[0]>source_binding['image_size'][0]:
            raise ValueError('mask crop extends outside original source')
        record=extract_mask(raw,score,identity)
        n,labels,stats,_=cv2.connectedComponentsWithStats((raw>0).astype(np.uint8),connectivity=8)
        components=[]
        for component in range(1,n):
            ys,xs=np.where(labels==component);points=np.column_stack((xs+translation[0],ys+translation[1],np.ones(len(xs))))
            hits={}
            for anchor_id,anchor in declared.items():
                hits[anchor_id]=0
                if not supported.get(anchor_id,False):continue
                q=points@np.asarray(pose_map[anchor_id]['inspection_to_reference_local']).T
                if np.any(np.abs(q[:,2])<1e-9) or not (np.all(q[:,2]>0) or np.all(q[:,2]<0)):
                    raise ValueError('local mapping horizon through native component')
                mapped=q[:,:2]/q[:,2:];a,b,c,d=anchor['bbox_xyxy']
                hits[anchor_id]=int(((mapped[:,0]>=a)&(mapped[:,0]<=c)&(mapped[:,1]>=b)&(mapped[:,1]<=d)).sum())
            good=bool(score>=.75 and not record['boundary_truncated'] and
                item.get('recipe')==POLICY['recipe'] and all(supported.get(k,False) for k in declared)
                and all(v>0 for v in hits.values()))
            if good:eligible.append({'record_id':identity,'component_index':component,'mask_array_sha256':record['mask_array_sha256']})
            if score>=.75 and hits[socket_id]>0:high_socket_touch=True
            components.append({'component_index':component,'pixel_count':int(stats[component,cv2.CC_STAT_AREA]),
                'anchor_pixel_support':hits,'eligible_visible_bundle_component':good})
        audit.append({'record_id':identity,'score':score,'mask_array_sha256':record['mask_array_sha256'],
            'whole_mask_geometry_state':record['geometry']['state'],
            'whole_mask_component_count':record['geometry']['component_count'],
            'whole_mask_branch_pixels':record['geometry']['branch_pixel_count'],
            'boundary_truncated':record['boundary_truncated'],'components':components})
    return {'kind':POLICY['kind'],'source_binding':source_binding,'expected_visible_attachment':scope['expected_visible_attachment'],
        'socket_state':socket_state,'local_anchor_proposals_supported':{k:supported.get(k,False) for k in declared},
        'sam_inventory_verified':sam_inventory_verified is True,'eligible_native_components':eligible,
        'unique_native_bundle_observation_supported':len(eligible)==1,'high_score_mask_touches_socket':high_socket_touch,
        'mask_audit':audit,'all_original_masks_and_components_retained':True,'mask_pixels_changed':False,
        'single_wire_contract_replaced':False,'electrical_continuity':'not_assessed','model_observer_count':1}


def compare_harness(scope,reference,inspection):
    confirmed=validate_scope(scope,scope['reference_binding'])
    for view in [reference,inspection]:
        if view.get('kind')!=POLICY['kind'] or view.get('expected_visible_attachment')!=scope['expected_visible_attachment']:
            raise ValueError('different evidence kind or scoped reference relation')
        validate_binding(view['source_binding'])
    if reference['source_binding']!=scope['reference_binding']:raise ValueError('reference photo binding differs')
    result={'decision':'insufficient_evidence','observation_granularity':'multiwire_bundle_only',
        'reference_review_confirmed':confirmed,'expected_visible_attachment':scope['expected_visible_attachment'],
        'single_wire_contract_replaced':False,'single_wire_connection_confirmed':False,
        'electrical_correctness':'not_assessed','electrical_continuity':'not_assessed',
        'physical_new_connections':0,'electrical_disconnections_confirmed':0,
        'model_observer_count':1,'not_field_accuracy_or_certified_inspection':True}
    reference_ready=(reference['sam_inventory_verified'] and reference['socket_state']=='mating_body_visible'
        and reference['unique_native_bundle_observation_supported'] and all(reference['local_anchor_proposals_supported'].values()))
    if not confirmed:result['reason']='reference_review_pending'
    elif not reference_ready:result['reason']='reference_visible_bundle_evidence_incomplete'
    elif inspection['socket_state']=='uncertain':result['reason']='socket_phenotype_uncertain'
    elif not inspection['sam_inventory_verified']:result['reason']='native_SAM_inventory_not_verified'
    else:
        socket_id=next(a['id'] for a in scope['anchors'] if a['kind']=='wire_entry_socket')
        if not inspection['local_anchor_proposals_supported'].get(socket_id,False):
            result['reason']='local_socket_proposal_unsupported'
        elif inspection['socket_state']=='socket_contacts_exposed':
            if inspection['high_score_mask_touches_socket']:
                result['reason']='exposed_socket_and_high_score_mask_conflict'
            else:
                result.update(decision='visible_socket_attachment_change_supported',
                    reason='positive_exposed_contact_phenotype_at_reference_socket_no_high_mask_contradiction',
                    fan_end_relationship_assessed=False)
        elif not all(inspection['local_anchor_proposals_supported'].values()):
            result['reason']='one_or_more_local_anchor_proposals_unsupported'
        elif not inspection['unique_native_bundle_observation_supported']:
            result['reason']='missing_or_ambiguous_native_two_anchor_bundle_component'
        else:
            result.update(decision='same_visible_bundle_attachment_supported',
                reason='visible_mating_body_and_unique_native_component_at_both_local_anchors',
                route_shape_not_compared=True,fan_end_relationship_assessed=True)
    return result
