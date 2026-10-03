import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
old=(ROOT/'artifacts/local_review_ui_v2_20261001/review.html').read_text(encoding='utf-8')
bundle=json.loads((ROOT/'artifacts/local_review_ui_v2_20261001/bundle.json').read_text(encoding='utf-8'))
maps={}
for case,version in bundle['source_versions'].items():
    path=ROOT/f'artifacts/source_entry_drafts_20261001/{case}_entry_draft.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=version['map_sha256']:raise ValueError('map version changed')
    mapping=json.loads(path.read_text(encoding='utf-8'))
    if mapping['image_binding']!=version['image_binding']:raise ValueError('map source changed')
    maps[case]=mapping
out=ROOT/'artifacts/local_review_ui_v3_20261002';out.mkdir(parents=True,exist_ok=True)
addon=(ROOT/'experiments/local_review_summary.js').read_text(encoding='utf-8')
page=old.replace('</script></html>','</script><script id="maps" type="application/json">'+json.dumps(maps,ensure_ascii=False).replace('<','\\u003c')+'</script><script>'+addon+'</script></html>')
page=page.replace('</style>','.queue-row{border-top:1px solid #cad4dc;padding:8px 0}.queue-row button{margin:4px;max-width:100%;overflow-wrap:anywhere}#review-summary{margin-bottom:24px}</style>')
(out/'review.html').write_text(page,encoding='utf-8')
(out/'bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
print('V3 built; V1/V2 preserved; review schema remains V2.')
