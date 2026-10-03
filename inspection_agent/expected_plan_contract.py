"""Versioned expected-plan drafts, deliberately ineligible for topology comparison."""
from copy import deepcopy
import hashlib
import json
import re

def revision_hash(doc):
    body={k:v for k,v in doc.items() if k!='revision_id'}
    return hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()

def make_draft(bundle,author,mode='operator_draft',parent=None):
    doc=dict(schema_version=1,kind='expected_plan_draft',bundle_id=bundle['bundle_id'],
             source_versions=deepcopy(bundle['source_versions']),author=author,record_mode=mode,
             status='draft_unconfirmed',confirmed=False,comparison_allowed=False,
             parent_revision_id=parent,content={'cases':[]})
    for case in sorted(bundle['source_versions']):
        doc['content']['cases'].append(dict(case=case,description='',selected_port_ids=[],port_labels={},expected_connections=None))
    doc['revision_id']=revision_hash(doc)
    return doc

def validate_draft(bundle,doc):
    keys={'schema_version','kind','bundle_id','source_versions','author','record_mode','status','confirmed','comparison_allowed','parent_revision_id','content','revision_id'}
    if not isinstance(doc,dict) or set(doc)!=keys or type(doc.get('schema_version')) is not int or doc['schema_version']!=1 or doc['kind']!='expected_plan_draft':raise ValueError('unsupported draft format')
    if doc['bundle_id']!=bundle['bundle_id'] or doc['source_versions']!=bundle['source_versions']:raise ValueError('draft source mismatch')
    if doc['status']!='draft_unconfirmed' or doc['confirmed'] is not False or doc['comparison_allowed'] is not False:raise ValueError('draft cannot confirm or allow comparison')
    if not isinstance(doc['author'],str) or not doc['author'].strip() or doc['record_mode'] not in ('operator_draft','automation_fixture'):raise ValueError('author and explicit record mode required')
    for key in ('revision_id','parent_revision_id'):
        v=doc[key]
        if v is None and key=='parent_revision_id':continue
        if not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v):raise ValueError('invalid revision hash')
    if doc['revision_id']!=revision_hash(doc):raise ValueError('revision content changed')
    content=doc['content']
    if not isinstance(content,dict) or set(content)!={'cases'} or not isinstance(content['cases'],list):raise ValueError('invalid draft content')
    seen=set()
    for row in content['cases']:
        if not isinstance(row,dict) or set(row)!={'case','description','selected_port_ids','port_labels','expected_connections'}:raise ValueError('invalid case fields')
        case=row['case']
        if not isinstance(case,str) or case not in bundle['source_versions'] or case in seen:raise ValueError('unknown/repeated case')
        seen.add(case)
        if not isinstance(row['description'],str):raise ValueError('description must be text')
        known={i['port_id'] for i in bundle['items'] if i['case']==case}
        ids=row['selected_port_ids']
        if not isinstance(ids,list) or any(not isinstance(i,str) or i not in known for i in ids) or len(ids)!=len(set(ids)):raise ValueError('unknown/repeated scope port')
        labels=row['port_labels']
        if not isinstance(labels,dict) or set(labels)!=set(ids):raise ValueError('labels must match selected scope')
        for label in labels.values():
            if not isinstance(label,dict) or set(label)!={'device_id','terminal_label'} or any(not isinstance(v,str) for v in label.values()):raise ValueError('invalid draft labels')
        edges=row['expected_connections']
        if edges is None:continue
        if not isinstance(edges,list):raise ValueError('expected connections must be null or list')
        pairs=set()
        for edge in edges:
            if not isinstance(edge,list) or len(edge)!=2 or any(not isinstance(p,str) or p not in ids for p in edge) or edge[0]==edge[1]:raise ValueError('edge outside scope or self connection')
            pair=tuple(sorted(edge))
            if pair in pairs:raise ValueError('duplicate expected connection')
            pairs.add(pair)
    if seen!=set(bundle['source_versions']):raise ValueError('missing cases')
    return dict(decision='expected_draft_recorded_not_confirmed',revision_id=doc['revision_id'],
                parent_revision_id=doc['parent_revision_id'],confirmed=False,comparison_allowed=False,
                connection_edges=[],automatic_fault_verdict=False)
