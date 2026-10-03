import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/local_review_ui_v2_20261001';OUT.mkdir(parents=True,exist_ok=True)
old=(ROOT/'artifacts/local_review_ui_20261001/review.html').read_text(encoding='utf-8')
encoded=re.search(r'<script id="bundle" type="application/json">(.*?)</script>',old,re.S).group(1)
bundle=json.loads(encoded)
(OUT/'bundle.json').write_text((ROOT/'artifacts/local_review_ui_20261001/bundle.json').read_text(encoding='utf-8'),encoding='utf-8')
template=(ROOT/'experiments/local_review_page.html').read_text(encoding='utf-8')
template=template[:template.index('<script id="bundle"')]
template=template.replace('局部證據復核','逐候選證據復核').replace('未處理 6 / 6；預設不作確認。刷新頁面會清除未匯出的填寫。','8 個候選均未處理。可保存草稿並匯入恢復；刷新仍會清除未匯出的內容。')
template=template.replace('<button id="preview">','<label>匯入已保存記錄<input id="import" type="file" accept=".json,application/json"></label><button id="apply" disabled>套用匯入內容（取代目前填寫）</button><button id="preview">')
template=template.replace('尚未回寫正式地圖或接入拓撲比較。','支持逐候選與舊版記錄恢復；尚未回寫正式地圖或接入拓撲比較。')
script=(ROOT/'experiments/local_review_v2.js').read_text(encoding='utf-8')
(OUT/'review.html').write_text(template+'<script id="bundle" type="application/json">'+encoded+'</script><script>'+script+'</script></html>',encoding='utf-8')
print('v2 built: 6 entries, 8 candidate reviews; v1 artifacts preserved')
