"""Bounded paired review experiment. Prepare locally first; --send explicit opt-in.

Manual per-image rectangles remove identifiers only, not detection tuning.
Human user authorized direct experimental transmission on 2026-10-09.
"""
import base64, hashlib, json, re, sys, time
from dataclasses import replace
from pathlib import Path
from prepare_private_review_comparison_20261009 import OUT, pending, CASES
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication
import private_region_review as private
import llm_review_priority as priority

# Conservative solid patches on label-bearing bands. Some useful pixels are
# intentionally hidden; the prompt treats them as unknown, never reconstructs.
PATCHES = {
    ('cabinet2','candidate_001'): [(0,0,999,36),(0,57,999,61),(0,195,999,999)],
    ('cabinet2','candidate_002'): [(0,0,999,70),(0,140,999,25),(0,177,999,999)],
    ('cabinet4_right1','candidate_001'): [(0,26,999,54),(0,110,999,999)],
    ('cabinet4_right1','candidate_002'): [(0,0,999,16),(0,33,999,39)],
    ('cabinet4_right1','candidate_003'): [(0,0,999,54),(0,91,999,999)],
}

def save_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare_all():
    staged=[]
    for name,source in CASES.items():
        p=pending(source); prepared=private.prepare(p); rows=[]
        for region in prepared['regions']:
            rendered=[]
            for im in region['images'][:2]:
                rects=[QRect(*v).intersected(im.rect()) for v in PATCHES[(name,region['candidate_id'])]]
                rendered.append(private.RedactionCanvas(im,rects).render())
            data=private.png(private.panel(region,*rendered))
            target=OUT/f'{name}_{region["candidate_id"]}_SEND_PREVIEW.png'
            target.write_bytes(data)
            rows.append({'candidate_id':region['candidate_id'],'png_base64':base64.b64encode(data).decode(),
                         'sha256':hashlib.sha256(data).hexdigest()})
        staged.append((name,source,p,prepared,rows))
    save_json(OUT/'privacy_staging.json',{
        'external_approval':'user explicitly authorized direct experimental sending on 2026-10-09',
        'visual_inspection_completed':False,'requests_performed':0,
        'manual_redactions_for_privacy_only':True,'redactions_hide_some_wire_context':True,
        'cases':[{'case':name,'source_binding':prepared['source_binding'],
                  'preview_sha256':[r['sha256'] for r in rows]} for name,_,_,prepared,rows in staged]})
    return staged

def send_all(staged, budget=2400, nonthinking=False):
    # Require separately inspected frozen bytes, rather than a preview GUI flag
    # silently standing in for user consent or actual image inspection.
    checked=json.loads((OUT/'privacy_inspection.json').read_text(encoding='utf-8'))
    if checked.get('all_five_previews_visually_inspected') is not True:
        raise ValueError('Local privacy inspection missing')
    expected={name:[r['sha256'] for r in rows] for name,_,_,_,rows in staged}
    if checked['preview_sha256']!=expected:raise ValueError('Inspected image bytes changed')
    folder='results_nonthinking' if nonthinking else ('results' if budget==2400 else f'results_budget{budget}')
    if (OUT/folder).exists():raise ValueError('Results exist; no duplicate requests')
    source=Path('C:/Users/HUAWEI/Desktop/中转站信息.txt').read_text(encoding='utf-8')
    urls=list(re.finditer(r'https://api\.deepseek\.com[^\s"\']*',source))
    if len(urls)!=1:raise ValueError('Ambiguous endpoint')
    keys=re.findall(r'sk-[A-Za-z0-9_-]+',source[max(0,urls[0].start()-300):urls[0].start()])
    if len(keys)!=1:raise ValueError('Ambiguous credential')
    token=keys[0]
    settings=replace(priority.base.backend.load_settings(),endpoint='https://api.deepseek.com/chat/completions',
                     max_tokens=budget,timeout_seconds=45)
    dest=OUT/folder;dest.mkdir();summaries=[]
    for name,src,p,prepared,rows in staged:
        packet={'version':1,'mode':'redacted_local_photos','operator_reviewed':True,
                'source_binding':prepared['source_binding'],'regions':rows}
        packet['packet_sha256']=private.digest(packet);private.validate(packet,p)
        for mode,visual in [('binary_masks',None),('redacted_local_photos',packet)]:
            before=private.sha(src); start=time.monotonic(); captured={}
            def sender(req,timeout):
                if nonthinking:
                    body=json.loads(req.data);body['thinking']={'type':'disabled'}
                    req=priority.Request(req.full_url,data=json.dumps(body,ensure_ascii=False).encode(),
                                         headers=dict(req.header_items()),method='POST')
                response=priority.base.backend._post_json(req,timeout)
                # Save only completion data, not auth, headers, request images.
                captured.update({k:response[k] for k in ('model','choices','usage') if k in response})
                return response
            result=priority.run(p['reference_mask'],p['inspection_mask'],p['candidates'],settings,token,
                                sender=sender,visual_packet=visual)
            result.update(case=name,elapsed_seconds=round(time.monotonic()-start,3),
                          output_budget=budget,
                          thinking_requested='disabled' if nonthinking else 'unchanged',
                          source_report_unchanged=before==private.sha(src),sam_rerun=False,
                          consent_basis='direct user authorization plus local visual redaction inspection')
            evidence=priority.local_evidence(p['report'],p['reference_mask'],p['inspection_mask'])
            result['priority_adjustment']=priority.adjust(result,evidence,
                priority.binding(p['reference_mask'],p['inspection_mask'],p['candidates']))
            encoded=json.dumps({'result':result,'completion':captured},ensure_ascii=False,indent=2)
            if token in encoded:raise ValueError('Secret unexpectedly echoed; refusing persistence')
            (dest/f'{name}_{mode}.json').write_text(encoded,encoding='utf-8')
            summaries.append({'case':name,'mode':mode,'status':result['status'],
                              'elapsed_seconds':result['elapsed_seconds'],'server_model':result['server_model'],
                              'diagnostic':result['diagnostic']})
            save_json(dest/'summary.json',{'calls_completed':len(summaries),'results':summaries,
                      'production_modified':False,'accuracy_gain_measured':False})
            print(json.dumps(summaries[-1],ensure_ascii=False),flush=True)

if __name__=='__main__':
    app=QApplication([]);fid=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    app.setFont(QFont(QFontDatabase.applicationFontFamilies(fid)[0],10))
    staged=prepare_all()
    if '--send' in sys.argv:send_all(staged,4096 if '--budget4096' in sys.argv else 2400,'--nonthinking' in sys.argv)
    else:print('Prepared five local-only redacted previews; no network requests.')
