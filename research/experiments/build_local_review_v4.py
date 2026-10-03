from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'artifacts/local_review_ui_v4_20261002';out.mkdir(parents=True,exist_ok=True)
old=(ROOT/'artifacts/local_review_ui_v3_20261002/review.html').read_text(encoding='utf-8')
addon=(ROOT/'experiments/local_expected_plan.js').read_text(encoding='utf-8')
page=old.replace('</script></html>','</script><script>'+addon+'</script></html>')
page=page.replace('</style>','#plan-editor input:not([type=checkbox]),#plan-editor textarea{display:block;box-sizing:border-box;max-width:100%;width:100%;margin:8px 0}#plan-editor{margin-bottom:24px;overflow-wrap:anywhere}#plan-editor button{margin:6px 4px;max-width:100%}</style>')
(out/'review.html').write_text(page,encoding='utf-8')
(out/'bundle.json').write_text((ROOT/'artifacts/local_review_ui_v3_20261002/bundle.json').read_text(encoding='utf-8'),encoding='utf-8')
print('V4 built; independent draft editor, V1/V2/V3 preserved.')
