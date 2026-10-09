"""Keep AI observations out of physical-identity GT and accuracy denominators."""
from collections import Counter

SOCKET_STATES={'mating_housing_apparent','contacts_exposed_apparent','unknown'}


def validate_draft(draft,manifest):
    if draft.get('reviewer_type')!='AI' or draft.get('human_confirmed') is not False:
        raise ValueError('This lane accepts explicitly provisional AI drafts only')
    for key in ['independent_physical_ground_truth','eligible_for_accuracy_scoring','eligible_for_threshold_calibration']:
        if draft.get(key) is not False:raise ValueError('AI draft cannot become ground truth or calibration labels')
    if draft.get('cross_photo_same_physical_wire_identity')!='unknown':
        raise ValueError('Photo-local observations do not certify cross-photo wire identity')
    ids={s['review_id'] for s in manifest['samples']};rows=draft['observations']
    if len(rows)!=len(ids) or {r['review_id'] for r in rows}!=ids:
        raise ValueError('Missing, duplicate or unbound review IDs')
    if any(r['socket'] not in SOCKET_STATES for r in rows):
        raise ValueError('Unsupported socket description')
    return dict(socket_observation_counts=dict(Counter(r['socket'] for r in rows)),
        case_count=len(rows),physical_identity_GT_count=0,human_confirmed_count=0,
        accuracy_denominator=0,eligible_for_model_calibration=False)


def compare_review(draft,manifest,endpoint_report):
    summary=validate_draft(draft,manifest)
    mapping={s['review_id']:s for s in manifest['samples']}
    old={r['id']:r for r in endpoint_report['cases']}
    rows=[]
    for observation in draft['observations']:
        binding=mapping[observation['review_id']];result=old[binding['case_id']]
        if binding['source_sha256']!=result['source_binding']['image_sha256']:
            raise ValueError('Source photo hash binding changed')
        exposed=observation['socket']=='contacts_exposed_apparent'
        hint=result['decision']=='reference_socket_exposure_observed'
        rows.append(dict(review_id=observation['review_id'],case_id=binding['case_id'],
            AI_socket_observation=observation['socket'],original_decision=result['decision'],
            original_reason=result['reason'],
            provisional_socket_difference_to_review=exposed and not hint,
            false_negative_confirmed=False,physical_identity_confirmed=False))
    return dict(summary,cases=rows,
        provisional_socket_difference_ids=[r['case_id'] for r in rows if r['provisional_socket_difference_to_review']],
        unchanged_original_result=True,accuracy_rate=None,field_accuracy=None)
