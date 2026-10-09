"""Native exact-anchor coverage only, separate from two-end route decisions."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from run_review import verified_run
from run_reference_color_paths_source import exact_regions
from local_socket_crop import native_coverage

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/local_socket_crop_source_20261008'


def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));verify(p['pins']);assert source_pins()==p['mainline_pins']
    inference=json.loads((OUT/'inference_report.json').read_text(encoding='utf-8'))
    assert inference['status']=='complete' and inference['protocol_sha256']==digest(OUT/'protocol.json')
    scope=json.loads(Path(p['confirmed_scope_path']).read_text(encoding='utf-8'));rows=[]
    prep=json.loads((ROOT/'artifacts/mendeley_cable_socket_controls_20261006/preparation_report.json').read_text(encoding='utf-8'))
    originals={r['id']:r for r in prep['cases']}
    for case in p['cases']:
        if case['id'] not in inference['completed']:continue
        inv=verified_run(OUT/case['id']/'cable',case['source']['path'])
        rgb=np.asarray(Image.open(case['source']['path']).convert('RGB'));raw=np.asarray(Image.open(case['original_source']['path']).convert('RGB').crop(case['crop_box_xyxy']))
        assert np.array_equal(rgb,raw)
        region=np.asarray(Image.open(case['CPU_region_path']).convert('L'))>0
        recomputed=exact_regions(rgb.shape[:2],case['crop_box_xyxy'],scope,case['anchors'])['FAN_CPU'];assert np.array_equal(region,recomputed)
        stats=[]
        for path,score in zip(inv['paths'],inv['scores']):
            mask=np.asarray(Image.open(path).convert('L'))>0
            rec=dict(record_id=path.stem,mask_sha256=digest(path),**native_coverage(rgb,region,mask,float(score)));stats.append(rec)
            display=rgb.copy();display[mask]=(display[mask]*.5+np.array([255,0,255])*.5).astype('uint8')
            canvas=Image.new('RGB',(max(rgb.shape[1],550),rgb.shape[0]+50),'white');canvas.paste(Image.fromarray(display),(0,0))
            d=ImageDraw.Draw(canvas);yy,xx=np.nonzero(region);d.rectangle((int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())),outline='lime')
            d.text((5,rgb.shape[0]+8),case['id']+' '+path.stem+' score='+str(float(score)),fill='black')
            d.text((5,rgb.shape[0]+28),'CPU warm/blue='+str(rec['exact_CPU_color_hits'])+'; NO connectivity verdict',fill='black')
            canvas.save(OUT/(case['id']+'_'+path.stem+'.png'))
        old=originals[case['id']];oldrgb=np.asarray(Image.open(old['crop_context']['source']['path']).convert('RGB'))
        oldregion=exact_regions(oldrgb.shape[:2],old['crop_context']['crop_box_xyxy'],scope,old['anchors'])['FAN_CPU'];baselines={}
        for recipe in ['cable','cable_plus_reference_anatomy_box']:
            oldinv=verified_run(Path(case['baseline_directory'])/recipe,old['crop_context']['source']['path'])
            obs=[dict(record_id=path.stem,**native_coverage(oldrgb,oldregion,np.asarray(Image.open(path).convert('L'))>0,float(score))) for path,score in zip(oldinv['paths'],oldinv['scores'])]
            baselines[recipe]=dict(records=obs,local_coverage=any(s['local_two_color_coverage_candidate'] for s in obs))
        rows.append(dict(id=case['id'],phenotype=case['phenotype'],records=stats,baselines=baselines,local_coverage=any(s['local_two_color_coverage_candidate'] for s in stats)))
    gains=[r['id'] for r in rows if r['local_coverage'] and not r['baselines']['cable']['local_coverage']]
    losses=[r['id'] for r in rows if not r['local_coverage'] and r['baselines']['cable']['local_coverage']]
    exposed=[r['id'] for r in rows if r['phenotype']=='socket_contacts_exposed' and r['local_coverage']]
    gate=len(rows)==5 and bool(gains) and not losses and not exposed and all(r['local_coverage'] for r in rows if r['phenotype']=='mating_body_visible')
    verify(p['pins']);assert source_pins()==p['mainline_pins']
    save(OUT/'report.json',dict(status='complete',cases=rows,gains=gains,losses=losses,exposed_local_candidates=exposed,local_coverage_gate=gate,
        fresh_encoders=inference['fresh_encoders'],model_observer_count=1,new_confirmed_connections=0,electrical_continuity='not_assessed',
        full_two_end_path_test_run=False,deployed=False,independent_audit='pending',actual_visual_review='pending'))
    print(json.dumps(dict(cases=len(rows),gains=gains,losses=losses,exposed=exposed,local_gate=gate)))


if __name__=='__main__':main()
