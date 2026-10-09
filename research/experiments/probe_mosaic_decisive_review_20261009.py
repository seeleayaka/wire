"""Paired 2x2 test: original/decisive prompt x opaque/mosaic privacy patches.

No change to local acceptance gates; eight requests maximum, no retries.
Mosaic is a deterministic inspection privacy transform, not reconstruction.
"""
import base64,hashlib,json,math,re,sys,time
from dataclasses import replace
from pathlib import Path
from prepare_private_review_comparison_20261009 import pending,CASES
from run_private_review_comparison_20261009 import PATCHES
from PyQt5.QtCore import QRect,Qt
from PyQt5.QtGui import QPainter,QFontDatabase,QFont
from PyQt5.QtWidgets import QApplication
import private_region_review as private
import llm_review_priority as priority

OUT=Path(__file__).resolve().parents[1]/'artifacts/mosaic_decisive_review_20261009'
DECISIVE=(
    '\nVisual review instructions: Make a specific decision on VISIBLE appearance, not an electrical verdict. '
    'The four panel labels are exact: A=reference PHOTO, B=inspection PHOTO, C=reference SAM MASK, D=inspection SAM MASK. '
    'C and D were computed independently BEFORE privacy processing and have NOT been blacked out or pixelated. '
    'Privacy patches in A/B therefore cannot cause missing pixels in C/D; never assert that relationship. '
    'Opaque or pixelated patches represent UNKNOWN photo detail, not physical obstruction, absent cables or faults. '
    'Inspect visible photo pixels OUTSIDE the privacy patches. If these show a distinct additional, absent or displaced '
    'wire-like object also supported by C/D, use visible_change with medium visual confidence; '
    'do not use insufficient_evidence solely because OTHER parts of the photo are redacted. '
    'If visible similarity and discrepancies support an alignment/segmentation artifact, use likely_artifact. '
    'Use insufficient_evidence only when the remaining visible evidence really cannot distinguish these alternatives. '
    'Write a DIFFERENT concrete observation per candidate: visible color where resolvable, shape and local relative position; '
    'never invent these details. Distinguish no corresponding curved structure from an entirely empty mask. '
    'High priority means a clear visible change deserves earlier human review, not an electrical fault. '
    'Do not assert the same cable identity or correct/incorrect electrical connection. '
    'Do not increase confidence or priority simply to satisfy this instruction. '
    'Each observation <=120 Chinese characters. Return only the original strict JSON schema. '
)
PHOTO_FIRST=(
    '\nPHOTO-FIRST evidence policy: Make a specific decision about visible appearance, never electrical continuity. '
    'A is the reference PHOTO and B the aligned inspection PHOTO. C and D are SAM reference/inspection MASKS. '
    'SAM is FALLIBLE: masks can omit real wires, include background, split one wire or change shape because of viewing angle. '
    'Trust clear unredacted/unpixelated photo pixels MORE than SAM mask shape or absence. '
    'When a wire is visible in a photo but absent from its mask, treat this as possible SAM error, not an absent physical wire. '
    'A mask discrepancy by itself does NOT establish visible physical change or justify medium confidence/high priority. '
    'C/D were generated independently BEFORE privacy processing and remain unchanged; privacy processing of A/B cannot cause C/D missingness. '
    'Opaque and pixelated photo patches contain UNKNOWN detail. Coarse mosaic color is NOT a resolved wire or verified geometry. '
    'Do not reconstruct hidden/photo-pixelated parts or use their absence as evidence. '
    'Use visible_change with medium confidence only if clear visible A/B photo pixels OUTSIDE privacy patches '
    'show a specific additional, missing or displaced wire-like structure. State that concrete photo evidence, '
    'its visible color if resolvable, its shape, and local relative position for EACH candidate. '
    'If photos look similar and the only discrepancy is SAM, prefer likely_artifact with low confidence, normal priority. '
    'If corresponding photo wire detail is hidden or unresolved, use insufficient_evidence, low confidence, normal priority '
    'even if masks differ strongly. Do not globally abstain when some unredacted photo evidence IS clear. '
    'Photographs can themselves suffer alignment, occlusion and perspective artifacts; do not infer cable identity or electrical correctness. '
    'High priority means earlier human visual review, not a fault. Never increase confidence to sound decisive. '
    'Each observation <=120 Chinese characters; return only the existing strict JSON schema. '
)

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def process(im,rects,style):
    if style=='opaque':return private.RedactionCanvas(im,rects).render()
    out=im.copy();p=QPainter(out)
    for rect in rects:
        if rect.isEmpty():continue
        crop=im.copy(rect)
        small=crop.scaled(max(1,rect.width()//18),max(1,rect.height()//18),Qt.IgnoreAspectRatio,Qt.SmoothTransformation)
        coarse=small.scaled(rect.size(),Qt.IgnoreAspectRatio,Qt.FastTransformation)
        p.drawImage(rect.topLeft(),coarse)
    p.end();return out

def prepare():
    OUT.mkdir(exist_ok=True);staged=[]
    for name,source in CASES.items():
        current=pending(source);prep=private.prepare(current)
        for style in ['opaque','mosaic']:
            rows=[];layouts=[]
            for region in prep['regions']:
                photos=[];layout=[]
                for im in region['images'][:2]:
                    rects=[QRect(*r).intersected(im.rect()) for r in PATCHES[(name,region['candidate_id'])]]
                    photos.append(process(im,rects,style))
                    layout.append([[round(r.x()/im.width(),4),round(r.y()/im.height(),4),
                                    round(r.width()/im.width(),4),round(r.height()/im.height(),4)] for r in rects])
                data=private.png(private.panel(region,*photos))
                (OUT/f'{name}_{region["candidate_id"]}_{style}.png').write_bytes(data)
                rows.append({'candidate_id':region['candidate_id'],'png_base64':base64.b64encode(data).decode(),
                             'sha256':hashlib.sha256(data).hexdigest()})
                layouts.append({'candidate_id':region['candidate_id'],'A_and_B_patch_xywh_normalized':layout})
            staged.append((name,style,source,current,prep,rows,layouts))
    manifest={f'{n}_{s}':[r['sha256'] for r in rows] for n,s,_,_,_,rows,_ in staged}
    save(OUT/'staging.json',{'preview_sha256':manifest,'requests_performed':0,
                          'source_bindings':{n:prep['source_binding'] for n,_,_,_,prep,_,_ in staged}})
    return staged,manifest

def send(staged,manifest,photo_first=False):
    inspected=json.loads((OUT/'privacy_inspection.json').read_text(encoding='utf-8'))
    if inspected.get('all_ten_previews_visually_inspected') is not True or inspected['preview_sha256']!=manifest:
        raise ValueError('Actual preview inspection missing or changed')
    dest=OUT/('results_photo_first' if photo_first else 'results')
    if dest.exists():raise ValueError('No duplicate runs permitted')
    secret=Path('C:/Users/HUAWEI/Desktop/中转站信息.txt').read_text(encoding='utf-8')
    urls=list(re.finditer(r'https://api\.deepseek\.com[^\s"\']*',secret))
    if len(urls)!=1:raise ValueError('Ambiguous official endpoint')
    keys=re.findall(r'sk-[A-Za-z0-9_-]+',secret[max(0,urls[0].start()-300):urls[0].start()])
    if len(keys)!=1:raise ValueError('Ambiguous adjacent credential')
    token=keys[0];settings=replace(priority.base.backend.load_settings(),endpoint='https://api.deepseek.com/chat/completions',
                                 timeout_seconds=45,max_tokens=2400)
    dest.mkdir();summary=[]
    for name,style,src,current,prep,rows,layouts in staged:
        packet={'version':1,'mode':'redacted_local_photos','operator_reviewed':True,
                'source_binding':prep['source_binding'],'regions':rows}
        packet['packet_sha256']=private.digest(packet);private.validate(packet,current)
        for prompt in (['photo_first'] if photo_first else ['original','decisive']):
            captured={};before=private.sha(src);start=time.monotonic()
            def sender(req,timeout):
                body=json.loads(req.data);body['thinking']={'type':'disabled'}
                if prompt in ('decisive','photo_first'):
                    body['messages'][1]['content'][0]['text']+=(PHOTO_FIRST if photo_first else DECISIVE)+' Photo privacy patch layout (local coordinates only): '+json.dumps(layouts)
                req=priority.Request(req.full_url,data=json.dumps(body,ensure_ascii=False).encode(),
                                     headers=dict(req.header_items()),method='POST')
                response=priority.base.backend._post_json(req,timeout)
                captured['model']=response.get('model');captured['usage']=response.get('usage')
                captured['choices']=[{'finish_reason':c.get('finish_reason'),
                     'message':{'content':c.get('message',{}).get('content')}} for c in response.get('choices',[])]
                return response
            result=priority.run(current['reference_mask'],current['inspection_mask'],current['candidates'],settings,token,
                                sender=sender,visual_packet=packet)
            result.update(case=name,privacy_style=style,prompt_variant=prompt,
                          elapsed_seconds=round(time.monotonic()-start,3),source_report_unchanged=before==private.sha(src),
                          thinking_requested='disabled',sam_rerun=False,mainline_modified=False)
            result['priority_adjustment']=priority.adjust(result,
                priority.local_evidence(current['report'],current['reference_mask'],current['inspection_mask']),
                priority.binding(current['reference_mask'],current['inspection_mask'],current['candidates']))
            value={'result':result,'completion':captured}
            if token in json.dumps(value,ensure_ascii=False):raise ValueError('Unexpected secret echo; refusing persistence')
            save(dest/f'{name}_{style}_{prompt}.json',value)
            item={'case':name,'privacy_style':style,'prompt':prompt,'status':result['status'],
                  'elapsed_seconds':result['elapsed_seconds'],
                  'judgments':[r['llm_judgment'] for r in result['priority_adjustment']['rows']],
                  'priorities':[r['adjusted_priority'] for r in result['priority_adjustment']['rows']]}
            summary.append(item);save(dest/'summary.json',{'calls_completed':len(summary),'maximum_calls':4 if photo_first else 8,'results':summary})
            print(json.dumps(item,ensure_ascii=False),flush=True)

if __name__=='__main__':
    app=QApplication([]);fid=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    app.setFont(QFont(QFontDatabase.applicationFontFamilies(fid)[0],10))
    staged,manifest=prepare()
    if '--send' in sys.argv:send(staged,manifest,'--photo-first' in sys.argv)
    else:print('Ten exact local previews prepared; no network requests.')
