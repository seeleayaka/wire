"""Independent raw polygon inverse mapping and source mask membership replay."""
import sys
from pathlib import Path
import json
import numpy as np
import cv2
from run_mask_label_ocr import prepare_cases
from audit_mask_label_ocr import independent_membership
from run_review import save,read
from run_audit import source_pins
from core import sha256,image_binding

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'artifacts/route_mask_axis_ocr_20261005'
OUT=ROOT/'artifacts/route_mask_axis_ocr_audit_20261005'


def main():
    if OUT.exists():raise FileExistsError('preserve affine audit')
    report=read(INPUT/'report.json');protocol=read(INPUT/'protocol.json')
    assert report['status']=='complete' and source_pins()==protocol['mainline_pins']
    assert all(sha256(p)==d for p,d in {**protocol['pins'],**protocol['models']}.items())
    counters=dict(raw_views=0,retained_polygon_inverse_replays=0,independent_membership_replays=0)
    results=[]
    for case in prepare_cases():
        folder=INPUT/case['id'];saved=read(folder/'report.json');size=case['full']['frame_binding']['image_size']
        assert image_binding(saved['image_path'])==saved['image_binding']
        by_id={r['record_id']:r for r in saved['fresh_readings']};rejected={r['record_id'] for r in saved['rejections']};seen=set()
        for path in sorted(folder.glob('mask_*_v*_raw.json')):
            raw=read(path);t=raw['transform'];turn=raw['quarter_turns'];w,h=t['output_size'];forward=np.array(t['source_crop_to_oriented'])
            inv=cv2.invertAffineTransform(forward)
            np.testing.assert_allclose(inv,t['oriented_to_source_crop'],rtol=0,atol=1e-12)
            counters['raw_views']+=1
            view=path.stem.removesuffix('_raw')
            for index,(points,text,score) in enumerate(raw['rows'] or []):
                rid='axis_'+view+f'_{index:03d}';seen.add(rid)
                assert (rid in by_id)!=(rid in rejected)
                if rid not in by_id:continue
                p=np.asarray(points,float);x=p[:,0].copy();y=p[:,1].copy()
                if turn==1:p=np.column_stack((2*w-1-y,x))
                elif turn==2:p=np.column_stack((2*w-1-x,2*h-1-y))
                elif turn==3:p=np.column_stack((y,2*h-1-x))
                p/=2
                source=p@inv[:,:2].T+inv[:,2]+np.array(t['source_crop_xyxy'][:2])
                np.testing.assert_allclose(source,by_id[rid]['polygon_source_xy'],rtol=0,atol=1e-10)
                assert by_id[rid]['text']==text and by_id[rid]['score']==float(score)
                counters['retained_polygon_inverse_replays']+=1
        assert seen==set(by_id)|rejected
        assert 4*sum(t['state']=='oriented_diagnostic_view' for t in saved['axis_transforms'])==len(list(folder.glob('mask_*_v*_raw.json')))
        high_counts={}
        for version,rows in [('fresh',saved['fresh_readings']),('combined',saved['old_raw_readings_preserved']+saved['fresh_readings'])]:
            members={r['record_id']:independent_membership(r['polygon_source_xy'],case['masks'],size) for r in rows}
            assert members==saved[version]['reading_memberships'];counters['independent_membership_replays']+=len(rows)
            high_counts[version]=sum(g['high_score_consistent_text_on_mask'] for g in saved[version]['groups'])
            assert all(g['confirmed'] is False and g['wire_identity'] is None and g['terminal_assignment'] is None and not g['text_identity_usable_automatically'] for g in saved[version]['groups'])
        results.append(dict(case=case['id'],high_consistent_text_on_mask=high_counts,confirmed_connections=0))
    assert counters['raw_views']==report['fresh_OCR_calls']==404
    OUT.mkdir();final=dict(status='pass',source_report_sha256=sha256(INPUT/'report.json'),auditor_sha256=sha256(__file__),
        counters=counters,cases=results,actual_new_overlays_inspected=True,no_new_inference=True,
        manual_review='Assistant qualitative review, not physical wire/terminal GT',no_deployment=True,field_accuracy=None)
    save(OUT/'report.json',final);print(json.dumps(final))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
