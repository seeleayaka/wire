"""Do not suppress a qualified socket observation behind a full-relation gate."""
from copy import deepcopy


def summarize_socket_observation(result,source_binding_verified):
    output=deepcopy(result)
    socket_supported=output['local_anchor_support'].get('FAN_CPU') is True
    state=output['socket_phenotype_observed']
    if source_binding_verified is not True or not socket_supported:
        observed='unknown';reason='source_or_local_socket_binding_not_verified'
    elif state=='socket_contacts_exposed':
        observed='socket_contacts_exposed_observed';reason='existing_frozen_socket_classifier_positive_observation'
    elif state=='mating_body_visible':
        observed='mating_housing_observed';reason='existing_frozen_socket_classifier_positive_observation'
    else:observed='unknown';reason='socket_classifier_abstained'
    conflict=output['socket_evidence_conflict'] is True
    output['socket_observation']=dict(state=observed,reason=reason,
        retained_despite_full_relation_unknown=observed!='unknown' and output['decision']=='insufficient_endpoint_evidence',
        native_mask_conflict=conflict,native_inventory_verified=output['native_inventory_verified'],
        disposition='manual_review_conflicting_evidence' if conflict else 'observation_only',
        full_relation_decision_unchanged=True,electrical_continuity='not_assessed',
        automatic_fault_confirmed=False,same_physical_wire_confirmed=False,
        existing_classifier_not_a_new_model=True)
    return output
