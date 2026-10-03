"""Candidate-wise review and explicit v1 migration; no inferred decisions."""
from collections import Counter
from copy import deepcopy
from .local_review_contract import validate_local_review


def migrate_v1(bundle, review):
    validate_local_review(bundle, review)
    old={row['item_id']:row for row in review['items']}
    result={k:deepcopy(v) for k,v in review.items() if k!='items'}
    result.update(schema_version=2,migrated_from_schema=1,items=[])
    for item in bundle['items']:
        row=old[item['item_id']]
        candidates=[]
        for rid in item['candidate_record_ids']:
            selected=row['record_id']==rid
            candidates.append(dict(record_id=rid,state=row['state'] if selected else 'pending',
                                   evidence_note=row['evidence_note'] if selected else ''))
        result['items'].append(dict(item_id=item['item_id'],bbox_xyxy=deepcopy(item['bbox_xyxy']),
            entry_review=dict(state='uncertain' if row['state']=='uncertain' and row['record_id'] is None else 'pending',
                              evidence_note=row['evidence_note'] if row['state']=='uncertain' and row['record_id'] is None else ''),
            candidate_reviews=candidates))
    return result


def validate_candidate_review(bundle, review):
    if not isinstance(review,dict) or type(review.get('schema_version')) is not int or review['schema_version']!=2:
        raise ValueError('v2 review schema required')
    if review.get('bundle_id')!=bundle['bundle_id'] or review.get('source_versions')!=bundle['source_versions']:
        raise ValueError('review source version mismatch')
    if review.get('review_mode') not in ('operator_review','automation_fixture') or not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
        raise ValueError('explicit mode and reviewer required')
    if review.get('connection_edges')!=[] or review.get('claim_scope')!='local_visible_relation_only':
        raise ValueError('local review cannot declare edges')
    expected={i['item_id']:i for i in bundle['items']};seen=set();supported=[];multi=[];summary=Counter()
    if not isinstance(review.get('items'),list):raise ValueError('items must be list')
    def check_state(row,entry=False):
        if not isinstance(row,dict):raise ValueError('review must be object')
        allowed=('pending','uncertain') if entry else ('pending','supported','rejected','uncertain')
        if row.get('state') not in allowed or not isinstance(row.get('evidence_note'),str):raise ValueError('invalid state/note')
        if (row['state']=='pending') != (not row['evidence_note'].strip()):raise ValueError('state and basis inconsistent')
    for row in review['items']:
        if not isinstance(row,dict) or not isinstance(row.get('item_id'),str):raise ValueError('invalid item')
        iid=row['item_id']
        if iid not in expected or iid in seen:raise ValueError('unknown/repeated item')
        seen.add(iid);item=expected[iid]
        if row.get('bbox_xyxy')!=item['bbox_xyxy']:raise ValueError('ROI changed')
        check_state(row.get('entry_review'),True)
        candidates=row.get('candidate_reviews')
        if not isinstance(candidates,list):raise ValueError('candidate reviews must be list')
        ids=set();local_support=[]
        for candidate in candidates:
            check_state(candidate);rid=candidate.get('record_id')
            if not isinstance(rid,str) or rid not in item['candidate_record_ids'] or rid in ids:raise ValueError('unknown/repeated candidate')
            ids.add(rid);summary[candidate['state']]+=1
            if candidate['state']=='supported':supported.append((item['case'],rid,iid));local_support.append(rid)
        if ids!=set(item['candidate_record_ids']):raise ValueError('missing candidate reviews')
        if len(local_support)>1:multi.append(iid)
    if seen!=set(expected):raise ValueError('missing entry reviews')
    usage=Counter((case,rid) for case,rid,_ in supported)
    return dict(decision='candidate_reviews_recorded_not_topology',review_mode=review['review_mode'],
        candidate_summary=dict(summary),shared_support_item_ids=sorted(set(iid for case,rid,iid in supported if usage[case,rid]>1)),
        multiple_supported_candidate_item_ids=multi,operator_review_verified=False,further_review_required=True,
        connection_edges=[],automatic_fault_verdict=False)
