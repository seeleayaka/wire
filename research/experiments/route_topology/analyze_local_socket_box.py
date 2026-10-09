"""Scalar audit and local-only contrast; no fragment/endpoint merging."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from run_review import verified_run
from audit_local_socket_crop import scalar_region,counts

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/local_socket_box_source_20261008'


def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));verify(p['pins']);assert source_pins()==p['mainline_pins']
    inf=json.loads((OUT/'inference_report.json').read_text(encoding='utf-8'));assert inf['status']=='complete' and inf['protocol_sha256']==digest(OUT/'protocol.json')
    scope=json.loads(Path(p['confirmed_scope_path']).read_text(encoding='utf-8'));anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU');rows=[]
    for case in p['cases']:
        if case['id'] not in inf['completed']:continue
        assert digest(case['original_source']['path'])==case['original_source']['image_sha256']
        rgb=np.asarray(Image.open(case['source']['path']).convert('RGB'));assert np.array_equal(rgb,np.asarray(Image.open(case['original_source']['path']).convert('RGB').crop(case['crop_box_xyxy'])))
        pose=next(a for a in case['anchors'] if a['id']=='FAN_CPU');region=scalar_region(rgb.shape[:2],case['crop_box_xyxy'],pose['inspection_to_reference_local'],anchor['bbox_xyxy'])
        assert np.array_equal(region,np.asarray(Image.open(case['CPU_region_path']).convert('L'))>0)
        crop=case['crop_box_xyxy'];box=case['positive_box_source_xyxy'];norm=case['positive_box_cxcywh_normalized'];w,h=rgb.shape[1],rgb.shape[0]
        # Independently map four expanded reference corners, not the preparation helper.
        l,t,r,b=anchor['bbox_xyxy'];pad=max(r-l,b-t)/2;inv=np.linalg.inv(np.asarray(pose['inspection_to_reference_local']));q=[]
        for x,y in [(l-pad,t-pad),(r+pad,t-pad),(r+pad,b+pad),(l-pad,b+pad)]:
            v=inv@np.array([x,y,1.]);q.append(v[:2]/v[2])
        expected=[int(np.floor(min(v[0] for v in q))),int(np.floor(min(v[1] for v in q))),int(np.ceil(max(v[0] for v in q))),int(np.ceil(max(v[1] for v in q)))]
        assert expected==box;np.testing.assert_allclose(norm,[(box[0]+box[2]-2*crop[0])/(2*w),(box[1]+box[3]-2*crop[1])/(2*h),(box[2]-box[0])/w,(box[3]-box[1])/h])
        recipes={}
        for recipe in p['recipes']:
            inventory=verified_run(OUT/case['id']/recipe,case['source']['path']);stats=[]
            for path,score in zip(inventory['paths'],inventory['scores']):
                mask=np.asarray(Image.open(path).convert('L'))>0;exact,total=counts(rgb,region,mask)
                rec=dict(record_id=path.stem,score=float(score),exact_CPU_color_hits=exact,context_color_hits=total,local_coverage=bool(score>=.75 and min(exact)>=8),mask_sha256=digest(path));stats.append(rec)
                display=rgb.copy();display[mask]=(display[mask]*.5+np.array([255,0,255])*.5).astype('uint8')
                canvas=Image.new('RGB',(550,h+50),'white');canvas.paste(Image.fromarray(display),(0,0));d=ImageDraw.Draw(canvas);yy,xx=np.nonzero(region)
                d.rectangle((int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())),outline='lime');d.text((5,h+8),case['id']+' '+recipe+' '+path.stem+' score='+str(float(score)),fill='black')
                d.text((5,h+28),'CPU warm/blue='+str(exact)+'; NO connectivity verdict',fill='black');canvas.save(OUT/(case['id']+'_'+recipe+'_'+path.stem+'.png'))
            recipes[recipe]=dict(records=stats,local_coverage=any(s['local_coverage'] for s in stats))
        rows.append(dict(id=case['id'],phenotype=case['phenotype'],recipes=recipes))
    old=lambda r:r['recipes']['cable']['local_coverage'];new=lambda r:r['recipes']['cable_plus_local_socket_box']['local_coverage']
    gains=[r['id'] for r in rows if new(r) and not old(r)];losses=[r['id'] for r in rows if old(r) and not new(r)];exposed=[r['id'] for r in rows if r['phenotype']=='socket_contacts_exposed' and new(r)]
    gate=len(rows)==5 and bool(gains) and not losses and not exposed and all(new(r) for r in rows if r['phenotype']=='mating_body_visible')
    verify(p['pins']);assert source_pins()==p['mainline_pins']
    save(OUT/'report.json',dict(status='complete',cases=rows,gains=gains,losses=losses,exposed_local_candidates=exposed,local_coverage_gate=gate,
        scalar_counts_and_coordinate_checks=True,HSV_conversion_shared=True,not_second_model=True,not_physical_identity_GT=True,
        fresh_encoders=inf['fresh_encoders'],new_confirmed_connections=0,full_two_end_path_test_run=False,deployed=False,visual_review='pending'))
    print(json.dumps(dict(cases=len(rows),gains=gains,losses=losses,exposed=exposed,local_gate=gate)))


if __name__=='__main__':main()
