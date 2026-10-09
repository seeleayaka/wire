"""Expanded trial preserves conflicting socket evidence explicitly."""
from endpoint_pair_candidate import compare_endpoint_pair as previous_compare


def compare_endpoint_pair(scope, reference, inspection):
    result=previous_compare(scope,reference,inspection)
    result['endpoint_candidate_schema_version']=2
    result['socket_phenotype_observed']=inspection['socket_state']
    result['socket_evidence_conflict']=bool(inspection['socket_state']=='socket_contacts_exposed'
                                          and inspection['high_score_mask_touches_socket'])
    if result['socket_evidence_conflict']:
        result.update(decision='insufficient_endpoint_evidence',
            reason='exposed_socket_and_high_score_mask_conflict',candidate_records=[])
    return result
