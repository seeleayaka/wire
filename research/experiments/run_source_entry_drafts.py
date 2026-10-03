"""Freeze source-only uncertain drafts then compare, never replace old maps."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
from inspection_agent.terminal_mapping import validate_mapping
from port_segment_review import review_port_contacts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/source_entry_drafts_20261001'
frozen = OUT / 'frozen_source_review.json'
digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
protocol = json.loads(frozen.read_text(encoding='utf-8'))
summary = []
for case in protocol['cases']:
    name = case['case']
    old_path = ROOT / f'artifacts/terminal_mapping_topology_20261001/{name}_draft.json'
    old_digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
    mapping = deepcopy(json.loads(old_path.read_text(encoding='utf-8')))
    if case.get('image_sha256') and case['image_sha256'] != mapping['image_binding']['image_sha256']:
        raise ValueError('source review image mismatch')
    mapping['map_id'] += '_source_review_v1'
    mapping['ports'] = [dict(p, device_id='source_strip_unconfirmed', terminal_label=p['id'],
        roi_kind='wire_entry_port', reviewer=None, evidence_note=p['uncertainty'], wire_label=None)
        for p in case['ports']]
    mapping['annotation_review'] = dict(method=protocol['review_method'], protocol_sha256=digest,
                                      independent_human_ground_truth=False)
    validate_mapping(mapping, Path(mapping['image_binding']['image_path']))
    # Source-only overlay is generated separately from any endpoint comparison.
    box = (305, 145, 365, 205) if name == 'cabinet_1' else (470, 175, 530, 215)
    with Image.open(mapping['image_binding']['image_path']) as image:
        plain = image.convert('RGB').crop(box).resize(((box[2]-box[0])*10, (box[3]-box[1])*10), Image.Resampling.NEAREST)
    overlay = plain.copy(); draw = ImageDraw.Draw(overlay)
    for p in mapping['ports']:
        x1,y1,x2,y2 = p['bbox_xyxy']
        draw.rectangle(((x1-box[0])*10,(y1-box[1])*10,(x2-box[0])*10-1,(y2-box[1])*10-1), outline='#ffb020',width=3)
        draw.text(((x1-box[0])*10,(y1-box[1])*10),p['id'],fill='red')
    sheet = Image.new('RGB',(plain.width*2,plain.height)); sheet.paste(plain,(0,0));sheet.paste(overlay,(plain.width,0))
    sheet.save(OUT / f'{name}_source_only_drafts.png')
    report = json.loads((ROOT / f'artifacts/sam_crop_coverage_20261001/{name}_source_endpoints.json').read_text(encoding='utf-8'))
    result = review_port_contacts(mapping, mapping['image_binding']['image_path'], report)
    per_port = []
    for port in mapping['ports']:
        contacts = [dict(record_id=row['record_id'], endpoint_index=row['endpoint_index'], point_xy=row['point_xy'],
                         state=row['state']) for row in result['rows'] if port['id'] in row['contact_port_ids']]
        diagnostics = [dict(record_id=row['record_id'], endpoint_index=row['endpoint_index'], point_xy=row['point_xy'],
                            distance_to_roi_closure_px=d['distance_to_roi_closure_px'])
                       for row in result['rows'] for d in row['nearest_diagnostics_not_assignments'] if d['port_id'] == port['id']]
        per_port.append(dict(port_id=port['id'], confirmed=False, contacts=contacts,
                             nearest_diagnostics_not_assignments=sorted(diagnostics,key=lambda d:(d['distance_to_roi_closure_px'],d['record_id']))[:3]))
    (OUT / f'{name}_per_port_audit.json').write_text(json.dumps(per_port,ensure_ascii=False,indent=2),encoding='utf-8')
    for filename, data in [(f'{name}_entry_draft.json',mapping),(f'{name}_contact_review.json',result)]:
        (OUT / filename).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == old_digest
    summary.append(dict(case=name, draft_ports=len(mapping['ports']), endpoint_states=result['summary'],
                        abstained_records=len(result['abstained_records']), independent_ground_truth=False,
                        automatic_connections_emitted=False, old_map_unchanged=True))
assert hashlib.sha256(frozen.read_bytes()).hexdigest() == digest
(OUT / 'summary.json').write_text(json.dumps(dict(protocol_sha256=digest,cases=summary),indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
