"""Original-byte/native-mask replay, not new SAM or blind validation."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_review import verified_run, save
from run_paired_evidence import load_run
from visible_bundle_relation import observe_bundle, compare_bundle

ROOT = Path(__file__).resolve().parents[2]

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def main():
    started=time.monotonic()
    base=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    recent=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    consolidated=ROOT/'artifacts/mendeley_semantic_bundle_consolidated_20261006/report.json'
    original=read(consolidated)
    if original['status']!='complete': raise ValueError('completed baseline required')
    before=source_pins()
    for p,digest in original['inputs'].items():
        if sha256(p)!=digest: raise ValueError('consolidated input drift')
    protocol=read(base/'protocol.json')
    scope=read(Path(protocol['confirmed_scope_path']))
    rows={r['id']:(base,r) for r in read(base/'preparation_report.json')['cases']}
    rows.update({r['id']:(recent,r) for r in read(recent/'preparation_report.json')['cases']})
    output=ROOT/'artifacts/mendeley_bundle_native_replay_20261008'
    output.mkdir(exist_ok=False)
    inputs=[consolidated,base/'protocol.json',base/'preparation_report.json',
            recent/'protocol.json',recent/'preparation_report.json',Path(protocol['confirmed_scope_path']),
            Path(__file__),Path(__file__).with_name('visible_bundle_relation.py')]
    pins={str(p):sha256(p) for p in inputs}
    save(output/'protocol.json',dict(pins=pins,mainline_pins=before,
        reuses_historical_masks=True,fresh_SAM=False,thresholds_unchanged=True,
        original_bytes_redecoded=True,not_blind_or_field_accuracy=True))
    def observe(identity):
        folder,row=rows[identity];source=row['original_source']
        if sha256(source['path'])!=source['image_sha256']: raise ValueError('original image drift')
        binding={k:source[k] for k in ['image_sha256','image_size','coordinate_frame']}
        context=row.get('crop_context');masks=[];verified=False
        if row['sam_inference_requested']:
            with Image.open(source['path']) as im:
                original_crop=np.asarray(im.convert('RGB').crop(context['crop_box_xyxy']))
            with Image.open(context['source']['path']) as im: cached=np.asarray(im.convert('RGB'))
            if not np.array_equal(original_crop,cached): raise ValueError('crop not exact original pixels')
            run=folder/identity/'cable_plus_reference_anatomy_box'
            origin,records=load_run(run);verified_run(run,origin['image_path'])
            for r in records:
                with Image.open(r['source_mask_path']) as im: raw=np.asarray(im.convert('L'))
                masks.append(dict(raw=raw,record_id=r['record_id'],score=r['score'],
                                  recipe='cable_plus_reference_anatomy_box'))
            verified=True
        observation=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],
            translation=tuple(context['crop_box_xyxy'][:2]) if context else (0,0),
            sam_inventory_verified=verified)
        return observation, source['path']
    reference,_=observe('reference')
    cases=[]
    for previous in original['cases']:
        observation,path=observe(previous['id'])
        result=compare_bundle(scope,reference,observation)
        if result!=previous['comparison']: raise ValueError('historical decision changed: '+previous['id'])
        category=Path(path).stem.rsplit('_',1)[0]
        cases.append(dict(id=previous['id'],dataset_category=category,
                          source_binding=observation['source_binding'],comparison=result,
                          native_mask_count=len(observation['mask_audit']),
                          native_two_anchor_components=len(observation['eligible_native_components'])))
        save(output/'progress.json',dict(status='replaying_native_masks',completed=len(cases),total=30))
    counts={name:sum(c['comparison']['decision']==name for c in cases)
            for name in sorted({c['comparison']['decision'] for c in cases})}
    reasons={name:sum(c['comparison']['reason']==name for c in cases)
             for name in sorted({c['comparison']['reason'] for c in cases if c['comparison']['decision']=='insufficient_evidence'})}
    if source_pins()!=before or any(sha256(p)!=digest for p,digest in pins.items()): raise ValueError('inputs/code/E drift')
    result=dict(status='complete',seconds=time.monotonic()-started,cases=cases,
        decision_counts=counts,unknown_reasons=reasons,old_decisions_reproduced=True,
        fresh_SAM_calls=0,automatic_new_support=0,electrical_connections_confirmed=0,
        scope='one_reviewed_fan_bundle_and_socket_not_whole_chassis',
        reused_development_data_not_field_accuracy=True,deployed=False)
    save(output/'report.json',result);save(output/'progress.json',dict(status='complete',completed=30,total=30))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))

if __name__=='__main__': main()
