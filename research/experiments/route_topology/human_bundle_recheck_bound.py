"""Standalone-report provenance adapter; no change to legacy active sessions."""
from copy import deepcopy
from human_bundle_recheck import recheck as legacy_recheck

def recheck(scope, case, submission):
    result = legacy_recheck(scope, case, submission)
    result['reference_binding'] = deepcopy(scope['reference_binding'])
    result['reference_anchor_declarations'] = deepcopy(scope['anchors'])
    result['reference_review'] = deepcopy(scope['reference_review'])
    result['schema_version'] = 2
    return result
