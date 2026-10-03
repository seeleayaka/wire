import json
from pathlib import Path
from port_segment_review import review_port_contacts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/port_contact_review_20261001'
OUT.mkdir(parents=True, exist_ok=True)
for name in ('cabinet_1', 'cabinet_2'):
    mapping = json.loads((ROOT / f'artifacts/terminal_mapping_topology_20261001/{name}_draft.json').read_text(encoding='utf-8'))
    report = json.loads((ROOT / f'artifacts/sam_crop_coverage_20261001/{name}_source_endpoints.json').read_text(encoding='utf-8'))
    result = review_port_contacts(mapping, mapping['image_binding']['image_path'], report)
    (OUT / f'{name}_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(name, json.dumps({'summary': result['summary'], 'abstained_records': len(result['abstained_records']),
                           'connection_edges': result['connection_edges']}))
