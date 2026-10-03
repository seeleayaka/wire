"""Live-bound local review attachment for Agent tasks, not a human fault verdict."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from .terminal_mapping import image_binding, validate_mapping
from .local_review_contract_v2 import migrate_v1, validate_candidate_review
from .expected_plan_contract import validate_draft

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def verify_evidence(bundle, files):
    metadata={k:v for k,v in bundle.items() if k!='bundle_id'}
    if hashlib.sha256(json.dumps(metadata,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=bundle.get('bundle_id'):
        raise ValueError('bundle content changed')
    if set(files)!=set(bundle['source_versions']):raise ValueError('evidence cases missing or unknown')
    seen=set()
    for case,version in bundle['source_versions'].items():
        paths=files[case]
        for key,name in [('map_sha256','map'),('endpoint_report_sha256','endpoints'),('audit_sha256','audit')]:
            if _sha(paths[name])!=version[key]:raise ValueError('live evidence version changed')
        mapping=_load(paths['map']);audit=_load(paths['audit'])
        actual=validate_mapping(mapping,Path(version['image_binding']['image_path']))
        if actual!=version['image_binding'] or mapping['image_binding']!=actual:raise ValueError('live source differs from bundle')
        if audit['entry_map_sha256']!=version['map_sha256'] or audit['image_binding']!=actual:raise ValueError('audit source differs')
        ports={p['id']:p for p in mapping['ports']};audits={p['port_id']:p for p in audit['ports']}
        items=[i for i in bundle['items'] if i['case']==case]
        if len(items)!=len(ports) or {i['port_id'] for i in items}!=set(ports):raise ValueError('bundle port scope changed')
        for item in items:
            iid=item['item_id'];pid=item['port_id'];ids=item['candidate_record_ids']
            if iid!=case+':'+pid or iid in seen or item['bbox_xyxy']!=ports[pid]['bbox_xyxy']:raise ValueError('bundle entry changed')
            seen.add(iid)
            if len(ids)!=len(set(ids)) or set(ids)!={c['record_id'] for c in audits[pid]['candidates']}:raise ValueError('bundle candidate set changed')
    if len(seen)!=len(bundle['items']):raise ValueError('unknown bundle case')

def prepare_attachment(task_report,bundle,review,files,case,*,plan=None,allow_test_records=False):
    if task_report['state']!='awaiting_human_review':raise ValueError('local review requires awaiting_human_review state')
    if case not in bundle['source_versions']:raise ValueError('unknown case')
    verify_evidence(bundle,files)
    actual=image_binding(Path(task_report['inputs']['inspection']))
    if actual!=bundle['source_versions'][case]['image_binding']:raise ValueError('task inspection does not match selected source case')
    normalized=migrate_v1(bundle,review) if review.get('schema_version')==1 else deepcopy(review)
    checked=validate_candidate_review(bundle,normalized)
    if normalized['review_mode']=='automation_fixture' and not allow_test_records:raise ValueError('test record rejected by default')
    if plan is not None:
        validate_draft(bundle,plan)
        if plan['record_mode']=='automation_fixture' and not allow_test_records:raise ValueError('test plan rejected by default')
    items=[i for i in bundle['items'] if i['case']==case];ids={i['item_id'] for i in items}
    rows=[r for r in normalized['items'] if r['item_id'] in ids]
    count=Counter(c['state'] for r in rows for c in r['candidate_reviews'])
    phase=task_report['machine_evidence'][-1]['phase']
    supplement=dict(schema_version=1,bundle_id=bundle['bundle_id'],case=case,phase=phase,
        visual_analysis_index=len(task_report['machine_evidence'])-1,
        image_binding=deepcopy(actual),source_versions=deepcopy(bundle['source_versions']),
        review_mode=normalized['review_mode'],reviewer=normalized['reviewer'],reviewer_authenticated=False,
        live_sources_verified=True,items=rows,candidate_summary=dict(count),
        shared_support_item_ids=[i for i in checked['shared_support_item_ids'] if i in ids],
        multiple_supported_candidate_item_ids=[i for i in checked['multiple_supported_candidate_item_ids'] if i in ids],
        expected_plan_draft=None,connection_edges=[],automatic_fault_verdict=False,
        claim_scope='local_visible_relation_only',human_fault_confirmation=False,
        topology_precheck=dict(decision='insufficient_evidence',comparison_performed=False,
            blockers=['local_opinions_not_verified_complete_cable_observations','port_identity_and_expected_table_require_separate_confirmation']))
    if plan is not None:
        supplement['expected_plan_draft']=dict(revision_id=plan['revision_id'],parent_revision_id=plan['parent_revision_id'],
            author=plan['author'],record_mode=plan['record_mode'],confirmed=False,comparison_allowed=False,
            content=deepcopy(next(c for c in plan['content']['cases'] if c['case']==case)))
    supplement['attachment_id']=hashlib.sha256(json.dumps(supplement,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return supplement

def attach_review_file(task_path,bundle_path,review_path,files,case,*,plan_path=None,allow_test_records=False):
    from .workflow import InspectionTask
    import os
    import tempfile
    target=Path(task_path)
    original=target.read_bytes()
    task=InspectionTask.from_report(json.loads(original.decode('utf-8')))
    result=task.record_local_evidence_review(_load(bundle_path),_load(review_path),files,case,
        plan=_load(plan_path) if plan_path is not None else None,allow_test_records=allow_test_records)
    if target.read_bytes()!=original:raise ValueError('task changed during import; retry from current task')
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=target.parent,
                                         prefix=target.name+'.local-review-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(task.to_report(),stream,ensure_ascii=False,indent=2)
            stream.flush();os.fsync(stream.fileno())
        if target.read_bytes()!=original:raise ValueError('task changed before save; retry')
        os.replace(temporary,target)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return result
