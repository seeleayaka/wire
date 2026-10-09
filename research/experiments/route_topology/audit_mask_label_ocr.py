"""Independent ROI polygon raster/count replay; no OCR or SAM rerun."""
import json
import hashlib
from pathlib import Path
import sys
import cv2
import numpy as np
from run_mask_label_ocr import prepare_cases,OUT as INPUT
from run_audit import source_pins
from run_review import save,read
from core import sha256,image_binding

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/route_mask_label_ocr_audit_20261005'


def independent_membership(polygon,masks,size):
    points=np.asarray(polygon,float);w,h=size
    x0=max(0,int(np.floor(points[:,0].min()))-1);y0=max(0,int(np.floor(points[:,1].min()))-1)
    x1=min(w,int(np.ceil(points[:,0].max()))+2);y1=min(h,int(np.ceil(points[:,1].max()))+2)
    raster=np.zeros((y1-y0,x1-x0),np.uint8)
    cv2.fillConvexPoly(raster,np.rint((points-[x0,y0])*256).astype(np.int32),1,shift=8)
    footprint=raster!=0;area=int(np.count_nonzero(footprint));groups={}
    for item in masks:
        binary=item['pixels']>0;digest=hashlib.sha256(binary.astype(np.uint8).tobytes()).hexdigest()
        if digest not in groups:
            groups[digest]=dict(id=item['id'],aliases=[],pixels=binary,eligible=item['topology_geometry_eligible'])
        g=groups[digest];g['aliases'].append(item['id']);g['eligible'] &= item['topology_geometry_eligible']
    supports=[]
    for g in groups.values():
        count=int(np.count_nonzero(footprint & g['pixels'][y0:y1,x0:x1]))
        if count:supports.append(dict(mask_id=g['id'],aliases=g['aliases'],intersection_pixels=count,
            text_polygon_fraction_on_mask=count/area,topology_geometry_eligible=g['eligible']))
    state=('ambiguous_multiple_masks' if len(supports)>1 else 'no_mask_overlap' if not supports else
           'insufficient_mask_overlap' if supports[0]['intersection_pixels']*2<area else 'unique_pixel_supported_nomination')
    return dict(state=state,supports=supports,nominated_mask_id=supports[0]['mask_id'] if state=='unique_pixel_supported_nomination' else None,
                confirmed_wire_identity=None,confirmed_port_identity=None)


def main():
    if OUT.exists():raise FileExistsError('preserve independent audit')
    protocol=read(INPUT/'protocol.json');report=read(INPUT/'report.json')
    assert report['status']=='complete' and report['fresh_local_OCR_calls']==408
    assert source_pins()==protocol['mainline_pins']
    assert all(sha256(p)==v for p,v in {**protocol['pins'],**protocol['models']}.items())
    OUT.mkdir();results=[];count=0
    for case in prepare_cases():
        case_report=read(INPUT/case['id']/'report.json')
        assert image_binding(case['full']['image_path'])==case_report['image_binding']
        assert case_report['baseline_raw_readings_preserved']==case['baseline']['records']
        fresh=case_report['fresh_readings'];old_reject={r['record_id'] for r in case_report['baseline_analysis_rejections']}
        baseline=[dict(r,record_id='base_'+r['record_id']) for r in case['baseline']['records'] if 'base_'+r['record_id'] not in old_reject]
        groups=[]
        for version,records in [('baseline',baseline),('fresh',fresh),('combined',baseline+fresh)]:
            by_id={r['record_id']:r for r in records};memberships={}
            for row in records:
                actual=independent_membership(row['polygon_source_xy'],case['masks'],case['full']['frame_binding']['image_size'])
                assert actual==case_report[version]['reading_memberships'][row['record_id']]
                memberships[row['record_id']]=actual;count+=1
            covered=[]
            for group in case_report[version]['groups']:
                tg=group['text_group'];ids=tg['record_ids'];covered.extend(ids)
                choices={memberships[i]['nominated_mask_id'] for i in ids}
                usable=len(choices)==1 and None not in choices
                high=usable and max(by_id[i]['score'] for i in ids)>=.9 and len({by_id[i]['text'].strip() for i in ids})==1
                assert group['high_score_consistent_text_on_mask']==high
                assert group['nominated_mask_id']==(next(iter(choices)) if usable else None)
                assert group['confirmed'] is False and group['wire_identity'] is None and group['terminal_assignment'] is None
                assert group['automatic_connections']==[] and not group['text_identity_usable_automatically']
                if version=='fresh' and usable:
                    groups.append(dict(mask=group['nominated_mask_id'],text=tg['best_text'],score=tg['best_score'],variants=tg['text_variants'],high=high))
            assert sorted(covered)==sorted(by_id)
        assert len(list((INPUT/case['id']).glob('mask_*_v*_raw.json')))==4*len(case['masks'])
        results.append(dict(case=case['id'],fresh_unique_groups=groups,high_score_attached=0,
                            raw_old_readings_preserved=True,confirmed_connections=0))
    final=dict(status='pass',input_report_sha256=sha256(INPUT/'report.json'),auditor_sha256=sha256(__file__),
               independent_ROI_polygon_membership_replays=count,cases=results,no_model_inference=True,
               manual_visual_review='Both fresh overlays inspected by assistant; not human GT or physical identity confirmation',
               E_unchanged=source_pins()==protocol['mainline_pins'],new_confirmed_connections=0,field_accuracy=None)
    save(OUT/'report.json',final);print(json.dumps(final,ensure_ascii=False))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
