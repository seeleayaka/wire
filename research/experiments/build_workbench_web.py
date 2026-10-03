import base64
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
old=(ROOT/'artifacts/local_review_ui_v4_20261002/review.html').read_text(encoding='utf-8')
scripts=old[old.index('<script id="bundle"'):old.rindex('</html>')]
bundle=json.loads((ROOT/'artifacts/local_review_ui_v4_20261002/bundle.json').read_text(encoding='utf-8'))
source=Path(next(iter(bundle['source_versions'].values()))['image_binding']['image_path'])
encoded='data:image/png;base64,'+base64.b64encode(source.read_bytes()).decode('ascii')
scripts+='<script id="source-original" type="text/plain">'+encoded+'</script>'
template=(ROOT/'experiments/workbench_shell.html').read_text(encoding='utf-8')
page=template.replace('__STYLE__',(ROOT/'experiments/workbench.css').read_text(encoding='utf-8')).replace('__SCRIPTS__',scripts).replace('__WORKBENCH_SCRIPT__',(ROOT/'experiments/workbench.js').read_text(encoding='utf-8'))
out=ROOT/'artifacts'/ (sys.argv[1] if len(sys.argv)>1 else 'wiremind_workbench_v2_20261002');out.mkdir(parents=True,exist_ok=True)
(out/'index.html').write_text(page,encoding='utf-8');(out/'bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
print('Self-contained web workbench built; existing review and plan contracts unchanged.')
