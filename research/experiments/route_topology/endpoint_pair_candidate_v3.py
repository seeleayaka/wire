"""Retain old single-socket observation without pretending two-end association."""
from endpoint_pair_candidate_v2 import compare_endpoint_pair as compare_v2


def compare_endpoint_pair(scope,reference,inspection):
    result=compare_v2(scope,reference,inspection)
    result['endpoint_candidate_schema_version']=3
    result['single_endpoint_observation_only']=False
    if result['automatic_topology_comparison']['decision']=='visible_socket_attachment_change_supported':
        result.update(decision='reference_socket_exposure_observed',
            reason='retain independently supported socket exposure; no two-end association claimed',
            candidate_records=[],single_endpoint_observation_only=True)
    return result
