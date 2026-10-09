"""Separate loopback companion; leaves the original review server and E untouched."""
import argparse
from http.server import HTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs, urlsplit
from core import image_binding, sha256
from review_guidance import build_plan, markdown
from serve_human_bundle_recheck import ROOT, handler, prepare

FRAGMENT = '''
<section class="panel" style="margin-top:18px;height:400px;overflow:auto"><h2>下一步复核</h2>
<p class="muted">依据已确认参考和当前证据生成；不调用大模型、不改变识别结论。</p>
<pre id="guidance" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></section>
<script>
(()=>{
const nativeFetch=window.fetch.bind(window), target=document.getElementById('guidance');
let epoch=0;
function render(plan,path){target.textContent=plan.label+'\\n\\n'+plan.actions.map(a=>a.instruction+'\\n依据：'+a.reason+'\\n状态：未执行，需要操作者。').join('\\n\\n')+'\\n\\n'+plan.limitations.join('\\n')+(path?'\\n\\n建议记录：'+path:'');}
async function initial(){const id=document.getElementById('case').value;if(!id)return;const n=++epoch;target.textContent='正在读取建议依据…';try{const r=await nativeFetch('/guidance?case='+encodeURIComponent(id));const p=await r.json();if(!r.ok)throw Error(p.error);if(n===epoch)render(p);}catch(e){if(n===epoch)target.textContent='建议不可用：'+e.message;}}
window.fetch=async(...args)=>{const r=await nativeFetch(...args);if(args[0]==='/recheck'&&r.ok){const data=await r.clone().json();++epoch;if(data.review_plan)render(data.review_plan,data.saved_review_plan);}return r;};
const select=document.getElementById('case');select.addEventListener('change',initial);
new MutationObserver(()=>{if(select.value)initial();}).observe(select,{childList:true});
const controls=document.querySelectorAll('input,textarea,select:not(#case)');
for(const c of controls)c.addEventListener(c.matches('input[type=checkbox],select')?'change':'input',initial);
document.getElementById('inspection').addEventListener('pointerup',initial);
initial();
})();
</script>
'''


def companion_handler(catalog, output, token, port):
    base = handler(catalog, output, token, port)

    class Companion(base):
        def do_GET(self):
            if urlsplit(self.path).path == '/guidance':
                if self.headers.get('Host') != '127.0.0.1:' + str(port):
                    return self.respond(403, {'error': 'invalid host'})
                try:
                    for p, digest in catalog['source_pins'].items():
                        if sha256(p) != digest:
                            raise ValueError('source evidence changed; reopen a new review session')
                    ids = parse_qs(urlsplit(self.path).query).get('case', [])
                    if len(ids) != 1:
                        raise ValueError('one case required')
                    case = next((c for c in catalog['cases'] if c['id'] == ids[0]), None)
                    if case is None:
                        raise ValueError('unknown case')
                    if (image_binding(case['source_path']) != case['image_binding']
                            or image_binding(output / ('images/' + case['id'] + '.jpg')) != case['image_binding']):
                        raise ValueError('inspection photo changed')
                    scope = catalog['scope']
                    if (image_binding(scope['reference_image_path']) != scope['reference_binding']
                            or image_binding(output / 'images/reference.jpg') != scope['reference_binding']):
                        raise ValueError('reference photo changed')
                    return self.respond(200, build_plan(catalog['scope'], case))
                except (ValueError, KeyError, TypeError) as e:
                    return self.respond(400, {'error': str(e)})
            return super().do_GET()

        def respond(self, status, value, mime='application/json; charset=utf-8'):
            if self.command == 'GET' and self.path == '/' and status == 200:
                value = value.replace(b'</html>', FRAGMENT.encode('utf-8') + b'</html>')
            if self.command == 'POST' and status == 200 and isinstance(value, dict) and 'report' in value:
                case = next(c for c in catalog['cases'] if c['id'] == value['report']['case_id'])
                plan = build_plan(catalog['scope'], case, value['report'])
                # Append a separate companion record. Never rewrite the evidence report.
                dest = Path(value['saved_report']).parent / 'guidance.json'
                plan['review_report_sha256'] = sha256(value['saved_report'])
                dest.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
                dest.with_suffix('.md').write_text(markdown(plan), encoding='utf-8')
                value = dict(value, review_plan=plan, saved_review_plan=str(dest))
            return super().respond(status, value, mime)
    return Companion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('new output must stay under workspace artifacts')
    catalog = prepare(output)
    for path in (Path(__file__), Path(__file__).with_name('review_guidance.py')):
        catalog['source_pins'][str(path)] = sha256(path)
    (output / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    server = HTTPServer(('127.0.0.1', args.port), companion_handler(catalog, output, secrets.token_hex(32), args.port))
    print('READY http://127.0.0.1:' + str(args.port), flush=True)
    server.serve_forever()


if __name__ == '__main__': main()
