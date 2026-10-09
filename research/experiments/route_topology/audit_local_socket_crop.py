"""Independent scalar coordinate/color count audit, not physical wire GT."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from run_review import verified_run

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/local_socket_crop_source_20261008'


def scalar_region(shape,crop,matrix,bbox):
    m=np.asarray(matrix);l,t,r,b=bbox;result=np.zeros(shape,bool)
    for y in range(shape[0]):
        for x in range(shape[1]):
            q=m@np.array([x+crop[0],y+crop[1],1.])
            if abs(q[2])<1e-9:raise ValueError('horizon')
            result[y,x]=l<=q[0]/q[2]<=r and t<=q[1]/q[2]<=b
    return result


def counts(rgb,region,mask):
    # HSV conversion shared with OpenCV; per-pixel family membership/counting independently scalar.
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);exact=[0,0];total=[0,0]
    for y,x in zip(*np.nonzero(mask)):
        hue,sat,val=[int(v) for v in hsv[y,x]]
        if sat<64 or val<32:continue
        group=0 if hue//10 in [0,1,2,3,16,17] else 1 if hue//10 in [9,10,11] else None
        if group is not None:
            total[group]+=1
            if region[y,x]:exact[group]+=1
    return exact,total


def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));verify(p['pins']);assert source_pins()==p['mainline_pins']
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    scope=json.loads(Path(p['confirmed_scope_path']).read_text(encoding='utf-8'));anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    observed={r['id']:r for r in report['cases']};oldprep=json.loads((ROOT/'artifacts/mendeley_cable_socket_controls_20261006/preparation_report.json').read_text(encoding='utf-8'))
    oldrows={r['id']:r for r in oldprep['cases']};regions=records=0;local={};oldlocal={}
    for case in p['cases']:
        assert digest(case['original_source']['path'])==case['original_source']['image_sha256']
        pose=next(a for a in case['anchors'] if a['id']=='FAN_CPU');inv=np.linalg.inv(np.asarray(pose['inspection_to_reference_local']))
        l,t,r,b=anchor['bbox_xyxy'];pad=max(r-l,b-t);points=[]
        for x,y in [(l-pad,t-pad),(r+pad,t-pad),(r+pad,b+pad),(l-pad,b+pad)]:
            q=inv@np.array([x,y,1.]);points.append([q[0]/q[2],q[1]/q[2]])
        expected=[math.floor(min(q[0] for q in points)),math.floor(min(q[1] for q in points)),math.ceil(max(q[0] for q in points)),math.ceil(max(q[1] for q in points))]
        assert expected==case['crop_box_xyxy']
        rgb=np.asarray(Image.open(case['source']['path']).convert('RGB'));original=np.asarray(Image.open(case['original_source']['path']).convert('RGB').crop(expected))
        assert np.array_equal(rgb,original)
        region=scalar_region(rgb.shape[:2],expected,pose['inspection_to_reference_local'],anchor['bbox_xyxy'])
        assert np.array_equal(region,np.asarray(Image.open(case['CPU_region_path']).convert('L'))>0);regions+=1
        if case['id'] not in observed:continue
        row=observed[case['id']];inventory=verified_run(OUT/case['id']/'cable',case['source']['path']);decisions=[]
        assert len(inventory['paths'])==len(row['records'])
        for path,score,record in zip(inventory['paths'],inventory['scores'],row['records']):
            assert record['record_id']==path.stem and record['mask_sha256']==digest(path) and record['score']==float(score)
            exact,total=counts(rgb,region,np.asarray(Image.open(path).convert('L'))>0)
            decision=bool(score>=.75 and min(exact)>=8)
            assert exact==record['exact_CPU_color_hits'] and total==record['context_color_hits']
            assert decision==record['local_two_color_coverage_candidate'];decisions.append(decision);records+=1
        local[case['id']]=any(decisions);assert local[case['id']]==row['local_coverage']
        old=oldrows[case['id']];oldrgb=np.asarray(Image.open(old['crop_context']['source']['path']).convert('RGB'))
        oldregion=scalar_region(oldrgb.shape[:2],old['crop_context']['crop_box_xyxy'],pose['inspection_to_reference_local'],anchor['bbox_xyxy'])
        for recipe in ['cable','cable_plus_reference_anatomy_box']:
            inventory=verified_run(Path(case['baseline_directory'])/recipe,old['crop_context']['source']['path']);decisions=[]
            saved=row['baselines'][recipe];assert len(inventory['paths'])==len(saved['records'])
            for path,score,record in zip(inventory['paths'],inventory['scores'],saved['records']):
                exact,total=counts(oldrgb,oldregion,np.asarray(Image.open(path).convert('L'))>0)
                decision=bool(score>=.75 and min(exact)>=8)
                assert exact==record['exact_CPU_color_hits'] and total==record['context_color_hits']
                assert decision==record['local_two_color_coverage_candidate'];decisions.append(decision);records+=1
            assert any(decisions)==saved['local_coverage']
            if recipe=='cable':oldlocal[case['id']]=any(decisions)
    gains=[k for k in local if local[k] and not oldlocal[k]];losses=[k for k in local if not local[k] and oldlocal[k]]
    exposed=[r['id'] for r in report['cases'] if r['phenotype']=='socket_contacts_exposed' and local[r['id']]]
    gate=len(local)==5 and bool(gains) and not losses and not exposed and all(local[r['id']] for r in report['cases'] if r['phenotype']=='mating_body_visible')
    assert gains==report['gains'] and losses==report['losses'] and exposed==report['exposed_local_candidates'] and gate==report['local_coverage_gate']
    verify(p['pins']);assert source_pins()==p['mainline_pins']
    result=dict(status='PASS',prepared_regions=regions,observed_cases=len(local),native_records_and_baselines=records,
        scalar_pixel_coordinates_and_counts=True,HSV_conversion_shared=True,not_physical_identity_GT=True,mainline_unchanged=True,
        pins={str(f):digest(f) for f in [Path(__file__),OUT/'report.json',OUT/'protocol.json',OUT/'inference_report.json']})
    save(OUT/'audit_report.json',result);print(json.dumps(result))


if __name__=='__main__':main()
