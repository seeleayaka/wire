"""Separate image-bound human evidence from immutable automatic observations."""
from copy import deepcopy
from datetime import datetime, timezone
import math
from core import text
from visible_lead_scope import validate_scope

SOURCE_LABELS = {'human_ui_review': '人工复核',
                 'software_fixture': '软件测试（非真实验收）',
                 'assistant_visual_review': 'AI 自查（非人类验收）'}
SOURCE_STATUS = {'human_ui_review': 'human_visual_review_recorded',
                 'software_fixture': 'software_fixture_recorded',
                 'assistant_visual_review': 'assistant_visual_review_recorded'}
RESULT_LABELS = {
    'human_confirmed_same_visible_attachment': '可见线束接法一致（走线路径不影响本结论）',
    'human_confirmed_changed_visible_attachment': '可见端点关系与参考不同',
    'human_confirmed_reference_socket_contacts_exposed': '参考插座接点露出；未判定电气断路',
    'insufficient_evidence': '补证仍不足，保留未知',
}


def recheck(scope, case, submission):
    if not isinstance(submission, dict):
        raise ValueError('review submission must be an object')
    if not validate_scope(scope, scope['reference_binding']):
        raise ValueError('normal reference has not been confirmed')
    if submission.get('case_id') != case['id'] or submission.get('image_binding') != case['image_binding']:
        raise ValueError('case/image fingerprint mismatch')
    reviewer = text(submission.get('reviewer'), 'reviewer')
    note = text(submission.get('evidence_note'), 'evidence note')
    submission_kind = submission.get('submission_kind')
    if submission_kind not in SOURCE_LABELS:
        raise ValueError('review provenance must be explicit')
    if submission.get('scope_acknowledged') is not True:
        raise ValueError('visible-bundle-only scope must be acknowledged')
    endpoints = submission.get('endpoints')
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError('exactly two endpoint records required')
    by_kind = {}
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            raise ValueError('endpoint must be an object')
        kind = endpoint.get('kind')
        if kind not in ('visible_lead_emergence', 'wire_entry_socket') or kind in by_kind:
            raise ValueError('one lead emergence and one socket required')
        normalized_identity = text(endpoint.get('identity'), 'endpoint identity')
        box = endpoint.get('bbox_xyxy')
        w, h = case['image_binding']['image_size']
        if (not isinstance(box, list) or len(box) != 4 or any(type(x) not in (float, int) or not math.isfinite(x) for x in box)
                or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
            raise ValueError('endpoint rectangle outside source image')
        if type(endpoint.get('identity_confirmed')) is not bool:
            raise ValueError('explicit endpoint identity review required')
        by_kind[kind] = dict(endpoint, identity=normalized_identity)
    a, b = [e['bbox_xyxy'] for e in endpoints]
    if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
        raise ValueError('endpoint rectangles overlap')
    state = submission.get('socket_state')
    if state not in ('attached', 'contacts_exposed', 'uncertain'):
        raise ValueError('unknown visible socket state')
    if type(submission.get('same_bundle_confirmed')) is not bool:
        raise ValueError('explicit bundle identity review required')
    route = submission.get('route_change')
    if route not in ('different', 'same', 'unknown'):
        raise ValueError('invalid route observation')
    expected = {a['kind']: text(a['id'], 'expected identity') for a in scope['anchors']}
    identities_ok = all(e['identity_confirmed'] for e in endpoints)
    same_ids = all(by_kind[k]['identity'] == expected[k] for k in expected)
    decision, reason = 'insufficient_evidence', 'human_identity_or_bundle_evidence_incomplete'
    if identities_ok and state == 'contacts_exposed' and by_kind['wire_entry_socket']['identity'] == expected['wire_entry_socket']:
        decision, reason = 'human_confirmed_reference_socket_contacts_exposed', 'visible socket state reviewed; not electrical disconnection'
    elif identities_ok and state == 'attached' and submission['same_bundle_confirmed']:
        decision = 'human_confirmed_same_visible_attachment' if same_ids else 'human_confirmed_changed_visible_attachment'
        reason = 'human-reviewed endpoint identities and bundle correspondence; route shape not used'
    return {
        'schema_version': 1, 'case_id': case['id'], 'created_utc': datetime.now(timezone.utc).isoformat(),
        'evidence_source': submission_kind, 'reviewer_identity_independently_authenticated': False,
        'review_source_label': SOURCE_LABELS[submission_kind],
        'review_result_label': SOURCE_LABELS[submission_kind] + '：' + RESULT_LABELS[decision],
        'reviewer': reviewer, 'evidence_note': note, 'image_binding': deepcopy(case['image_binding']),
        'automatic_comparison_unchanged': deepcopy(case['automatic_comparison']),
        'human_review': deepcopy(submission), 'human_recheck_decision': decision, 'reason': reason,
        'expected_visible_attachment': scope['expected_visible_attachment'],
        'observed_visible_attachment': {k: by_kind[k]['identity'] for k in by_kind}
            if identities_ok and state == 'attached' and submission['same_bundle_confirmed'] else None,
        # Keep the legacy decision field for existing readers. Its identifier is
        # not a human attestation; source/status/labels declare actual provenance.
        'recheck_execution_status': SOURCE_STATUS[submission_kind],
        'fresh_SAM_inference': False, 'automatic_new_hits': 0,
        'electrical_correctness': 'not_assessed', 'electrical_continuity': 'not_assessed',
        'single_wire_identity_confirmed': False, 'physical_repair_performed': False,
        'deployed_to_E_mainline': False,
    }
