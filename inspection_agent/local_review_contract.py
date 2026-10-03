"""Version-bound local human-review records; never emit topology edges."""
from collections import Counter


def validate_local_review(bundle, review):
    if not isinstance(review,dict) or type(review.get('schema_version')) is not int or review['schema_version']!=1:
        raise ValueError('invalid review schema')
    if review.get('bundle_id')!=bundle['bundle_id'] or review.get('source_versions')!=bundle['source_versions']:
        raise ValueError('review source version mismatch')
    mode=review.get('review_mode')
    if mode not in ('operator_review','automation_fixture'):
        raise ValueError('explicit review mode required')
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
        raise ValueError('reviewer required')
    if review.get('connection_edges')!=[] or review.get('claim_scope')!='local_visible_relation_only':
        raise ValueError('local review cannot declare connection edges')
    expected={item['item_id']:item for item in bundle['items']}
    rows=review.get('items');seen=set();supported=[];states=Counter()
    if not isinstance(rows,list):raise ValueError('review items must be a list')
    for row in rows:
        if not isinstance(row,dict):raise ValueError('review item must be object')
        item_id=row.get('item_id')
        if item_id not in expected or item_id in seen:raise ValueError('unknown or repeated item')
        seen.add(item_id);item=expected[item_id]
        if row.get('bbox_xyxy')!=item['bbox_xyxy']:raise ValueError('entry ROI changed')
        state=row.get('state')
        if state not in ('pending','supported','rejected','uncertain'):raise ValueError('invalid review state')
        rid=row.get('record_id')
        if rid is not None and rid not in item['candidate_record_ids']:raise ValueError('unknown candidate record')
        note=row.get('evidence_note')
        if not isinstance(note,str):raise ValueError('note must be text')
        if state!='pending' and not note.strip():raise ValueError('review basis required')
        if state in ('supported','rejected') and rid is None:raise ValueError('candidate required')
        if state=='pending' and (rid is not None or note.strip()):raise ValueError('pending item contains review')
        if state=='supported':supported.append((item['case'],rid,item_id))
        states[state]+=1
    if seen!=set(expected):raise ValueError('review must preserve all items including pending')
    counts=Counter((case,rid) for case,rid,_ in supported)
    conflicts=[iid for case,rid,iid in supported if counts[case,rid]>1]
    return dict(decision='local_review_recorded_not_topology',review_mode=mode,
                operator_review_verified=False,summary=dict(states),shared_support_item_ids=conflicts,
                further_review_required=True,connection_edges=[],automatic_fault_verdict=False)
