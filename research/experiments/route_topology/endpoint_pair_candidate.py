"""Record-level endpoint correspondence; explicitly NOT a topology edge."""
from copy import deepcopy
from visible_bundle_relation import compare_bundle


def compare_endpoint_pair(scope, reference, inspection):
    original = compare_bundle(scope,reference,inspection)
    result = dict(decision='insufficient_endpoint_evidence',reason='endpoint_evidence_incomplete',
        automatic_topology_comparison=deepcopy(original),
        reference_bundle_record_id=list(scope['expected_visible_attachment']),
        candidate_records=[],middle_connection='unobserved',
        same_physical_wire_confirmed=False,single_wire_identity_confirmed=False,
        electrical_continuity='not_assessed',electrical_correctness='not_assessed',
        physical_new_connections=0,mask_pixels_changed=False,model_observer_count=1,
        observation_granularity='reference_multiwire_bundle_endpoint_pair_candidate',
        deployed=False)
    reference_ready=(original['reference_review_confirmed'] and reference['sam_inventory_verified']
        and reference['socket_state']=='mating_body_visible'
        and reference['unique_native_bundle_observation_supported']
        and all(reference['local_anchor_proposals_supported'].values()))
    if not reference_ready:
        result['reason']='reference_not_ready'
        return result
    if not inspection['sam_inventory_verified']:
        result['reason']='native_inventory_not_verified'
        return result
    if not all(inspection['local_anchor_proposals_supported'].values()):
        result['reason']='one_or_more_endpoint_local_correspondences_unsupported'
        return result
    if inspection['socket_state']=='socket_contacts_exposed':
        result.update(decision='reference_socket_exposure_observed',
                      reason='positive socket phenotype; not electrical disconnection')
        return result
    if inspection['socket_state']!='mating_body_visible':
        result['reason']='socket_state_uncertain'
        return result
    anchors=[a['id'] for a in scope['anchors']]
    candidates=[]
    for mask in inspection['mask_audit']:
        if mask['score']<.75 or mask['boundary_truncated']:continue
        support={a:[c['component_index'] for c in mask['components'] if c['anchor_pixel_support'][a]>0]
                 for a in anchors}
        if all(support.values()):
            candidates.append(dict(record_id=mask['record_id'],score=mask['score'],
                anchor_components=support,mask_array_sha256=mask['mask_array_sha256'],
                whole_native_component_count=mask['whole_mask_component_count']))
    result['candidate_records']=candidates
    if len(candidates)!=1:
        result['reason']='missing_or_ambiguous_same_instance_endpoint_support'
        return result
    result.update(decision='reference_endpoint_pair_candidate',
        reason='both registered endpoints supported by one SAM instance; physical identity unconfirmed',
        middle_connection=('native_visible_component_present' if inspection['unique_native_bundle_observation_supported']
                           else 'unobserved'))
    return result
