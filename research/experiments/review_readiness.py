"""Turn local opinions into a recheck queue, never into connection edges."""
from collections import Counter
from local_review_contract_v2 import migrate_v1, validate_candidate_review


def summarize_review(bundle, review, mappings):
    if review.get('schema_version') == 1:
        review = migrate_v1(bundle, review)
    checked = validate_candidate_review(bundle, review)
    conflicts = set(checked['shared_support_item_ids'] + checked['multiple_supported_candidate_item_ids'])
    entries = {r['item_id']: r for r in review['items']}
    rows = []
    for item in bundle['items']:
        row = entries[item['item_id']]
        states = Counter(c['state'] for c in row['candidate_reviews'])
        reasons = []
        if item['item_id'] in conflicts: reasons.append('conflicting_support')
        if not row['candidate_reviews']: reasons.append('no_visible_candidate')
        if states['pending']: reasons.append('pending_candidates')
        if states['uncertain'] or row['entry_review']['state'] == 'uncertain': reasons.append('uncertain_evidence')
        if row['candidate_reviews'] and states['rejected'] == len(row['candidate_reviews']): reasons.append('all_candidates_rejected_not_missing_wire')
        if states['supported']: reasons.append('local_support_not_connection')
        mapping = mappings.get(item['case'], {})
        ports = [p for p in mapping.get('ports', []) if p.get('id') == item['port_id']]
        if len(ports) != 1 or ports[0].get('confirmed') is not True:
            reasons.append('port_identity_unconfirmed')
        elif ports[0].get('bbox_xyxy') != item['bbox_xyxy']:
            raise ValueError('confirmed port ROI differs from review bundle')
        rows.append(dict(item_id=item['item_id'], title=item['title'], case=item['case'],
                         candidate_counts={s:states[s] for s in ('supported','rejected','uncertain','pending')},
                         candidate_reviews=row['candidate_reviews'], entry_review=row['entry_review'],
                         reasons=reasons, needs_recheck=True))
    case_gates = []
    for case in sorted({i['case'] for i in bundle['items']}):
        mapping = mappings.get(case)
        blockers = ['no_verified_complete_cable_observations']
        if review['review_mode'] == 'automation_fixture': blockers.append('test_record_not_operator_evidence')
        if mapping is None: blockers.append('missing_port_map')
        else:
            if mapping.get('image_binding') != bundle['source_versions'][case]['image_binding']:
                raise ValueError('map source differs from bundle')
            case_rows = [r for r in rows if r['case'] == case]
            if any('port_identity_unconfirmed' in r['reasons'] for r in case_rows): blockers.append('port_identity_unconfirmed')
            if mapping.get('scope',{}).get('expected_complete') is not True: blockers.append('expected_scope_unconfirmed')
            if not isinstance(mapping.get('expected_connections'),list): blockers.append('expected_connections_unknown')
            expected = mapping.get('expected_review',{})
            if expected.get('confirmed') is not True or not all(isinstance(expected.get(k),str) and expected[k].strip() for k in ('reviewer','evidence_note')):
                blockers.append('expected_table_unconfirmed')
        if any(r['item_id'] in conflicts for r in rows if r['case']==case): blockers.append('conflicting_local_support')
        case_gates.append(dict(case=case,decision='insufficient_evidence',blockers=blockers,
                               comparison_performed=False,connection_edges=[]))
    return dict(schema_version=1,bundle_id=bundle['bundle_id'],source_versions=bundle['source_versions'],
                review_mode=review['review_mode'],reviewer=review['reviewer'],candidate_summary=checked['candidate_summary'],
                entries=rows,topology_readiness=case_gates,connection_edges=[],automatic_fault_verdict=False,
                claim_scope='review_queue_and_preconditions_only')
