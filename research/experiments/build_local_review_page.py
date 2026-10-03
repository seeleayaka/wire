import base64
import hashlib
import json
from pathlib import Path
from inspection_agent.terminal_mapping import validate_mapping

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/local_review_ui_20261001';OUT.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=lambda p:'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode('ascii')
items=[];versions={}
reasons={'shared_segment_across_ports_ambiguous':'同一線段被多個入口共享，不能自動分配。',
         'multiple_pixel_paths_ambiguous':'有多條像素路徑；相鄰導線或掩膜覆蓋可能造成歧義。',
         'single_pixel_path_unconfirmed':'只有一個像素路徑候選，仍需確認可見關係。',
         'no_continuous_pixel_support_not_missing_wire':'未找到連續像素支持，不能判缺線。'}
for case in ('cabinet_1','cabinet_2'):
    map_path=ROOT/f'artifacts/source_entry_drafts_20261001/{case}_entry_draft.json'
    ep_path=ROOT/f'artifacts/sam_crop_coverage_20261001/{case}_source_endpoints.json'
    audit_path=ROOT/f'artifacts/directional_entry_path_20261001/{case}_audit.json'
    mapping=load(map_path);audit=load(audit_path);binding=validate_mapping(mapping,Path(mapping['image_binding']['image_path']))
    assert audit['entry_map_sha256']==sha(map_path)
    assert audit['image_binding']['image_sha256']==binding['image_sha256']
    versions[case]=dict(image_binding=binding,map_sha256=sha(map_path),endpoint_report_sha256=sha(ep_path),audit_sha256=sha(audit_path))
    for i,port in enumerate(mapping['ports']):
        evidence=next(p for p in audit['ports'] if p['port_id']==port['id'])
        ids=[c['record_id'] for c in evidence['candidates']]
        items.append(dict(item_id=case+':'+port['id'],case=case,port_id=port['id'],bbox_xyxy=port['bbox_xyxy'],
            title=f'視角{1 if case=="cabinet_1" else 2} · 入口{i+1}',reason=reasons[evidence['state']],
            candidate_record_ids=ids,source_image=data(ROOT/f'artifacts/source_entry_drafts_20261001/{case}_source_only_drafts.png'),
            candidate_images={rid:data(ROOT/f'artifacts/directional_entry_path_20261001/{case}_{port["id"]}_{rid}.png') for rid in ids}))
metadata=dict(schema_version=1,source_versions=versions,items=[{k:v for k,v in item.items() if k not in ('source_image','candidate_images')} for item in items])
bundle_id=hashlib.sha256(json.dumps(metadata,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
bundle=dict(metadata,bundle_id=bundle_id);(OUT/'bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
page_data=dict(bundle,items=items)
template=(ROOT/'experiments/local_review_page.html').read_text(encoding='utf-8')
page=template.replace('__BUNDLE__',json.dumps(page_data,ensure_ascii=False).replace('<','\\u003c'))
(OUT/'review.html').write_text(page,encoding='utf-8')
print(json.dumps(dict(items=len(items),bundle_id=bundle_id,page_bytes=len(page.encode())),indent=2))
