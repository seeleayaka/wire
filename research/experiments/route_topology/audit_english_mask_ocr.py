"""Independent native-crop raw-coordinate and mask membership replay."""
from pathlib import Path
import sys
import json
import numpy as np
from run_mask_label_ocr import prepare_cases
from audit_mask_label_ocr import independent_membership
from run_audit import source_pins
from run_review import read,save
from core import sha256,image_binding

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'artifacts/route_english_mask_ocr_20261005'
OUT=ROOT/'artifacts/route_english_mask_ocr_audit_20261005'


def main():
    if OUT.exists():raise FileExistsError('preserve native English OCR audit')
    report=read(INPUT/'report.json');protocol=read(INPUT/'protocol.json')
    assert report['status']=='complete' and source_pins()==protocol['mainline_pins']
    assert all(sha256(p)==d for p,d in {**protocol['pins'],**protocol['old_model_pins']}.items())
    totals=dict(raw_view_replays=0,retained_polygon_replays=0,membership_replays=0);cases=[]
    for case in prepare_cases():
        folder=INPUT/case['id'];saved=read(folder/'report.json');size=case['full']['frame_binding']['image_size']
        assert image_binding(saved['image_path'])==saved['image_binding']
        by_id={r['record_id']:r for r in saved['fresh_readings']};reject={r['record_id'] for r in saved['rejections']};seen=set()
        for path in sorted(folder.glob('mask_*_v*_raw.json')):
            raw=read(path);box=raw['source_crop_xyxy'];w=2*(box[2]-box[0]);h=2*(box[3]-box[1])
            view=path.stem.removesuffix('_raw');turn=int(view.rsplit('_v',1)[1]);totals['raw_view_replays']+=1
            assert raw['recognizer_sha256']==protocol['recognizer_sha256']
            for i,(polygon,text,score) in enumerate(raw['rows'] or []):
                rid='eng5_'+view+f'_{i:03d}';seen.add(rid)
                assert (rid in by_id)!=(rid in reject)
                if rid not in by_id:continue
                p=np.asarray(polygon,float);x=p[:,0].copy();y=p[:,1].copy()
                if turn==1:p=np.column_stack((w-1-y,x))
                elif turn==2:p=np.column_stack((w-1-x,h-1-y))
                elif turn==3:p=np.column_stack((y,h-1-x))
                p=p/2+np.array(box[:2]);row=by_id[rid]
                np.testing.assert_allclose(p,row['polygon_source_xy'],rtol=0,atol=1e-10)
                assert row['text']==text and row['score']==float(score) and row['recognizer_sha256']==protocol['recognizer_sha256']
                totals['retained_polygon_replays']+=1
        assert seen==set(by_id)|reject
        assert len(list(folder.glob('mask_*_v*_raw.json')))==4*len(case['masks'])
        summary={}
        for version,rows in [('fresh',saved['fresh_readings']),('combined',saved['old_raw_readings_preserved']+saved['fresh_readings'])]:
            members={r['record_id']:independent_membership(r['polygon_source_xy'],case['masks'],size) for r in rows}
            assert members==saved[version]['reading_memberships'];totals['membership_replays']+=len(rows)
            by={r['record_id']:r for r in rows};high=[]
            for g in saved[version]['groups']:
                ids=g['text_group']['record_ids'];choices={members[i]['nominated_mask_id'] for i in ids}
                usable=len(choices)==1 and None not in choices
                consistent=usable and max(by[i]['score'] for i in ids)>=.9 and len({by[i]['text'].strip() for i in ids})==1
                assert consistent==g['high_score_consistent_text_on_mask']
                assert not g['confirmed'] and g['wire_identity'] is None and g['terminal_assignment'] is None and g['automatic_connections']==[]
                if consistent:high.append(dict(text=g['text_group']['best_text'],score=g['text_group']['best_score'],mask=g['nominated_mask_id']))
            summary[version]=dict(high_consistent_nomination_count=len(high),nominations=high)
        cases.append(dict(case=case['id'],**summary,confirmed_physical_identities=0,confirmed_connections=0))
    assert totals['raw_view_replays']==report['fresh_OCR_calls']==408
    OUT.mkdir();final=dict(status='pass',source_report_sha256=sha256(INPUT/'report.json'),auditor_sha256=sha256(__file__),
        counters=totals,cases=cases,no_new_inference=True,no_deployment=True,field_accuracy=None,
        warning='Numeric OCR confidences are not calibrated physical identity or electrical connection truth.')
    save(OUT/'report.json',final);print(json.dumps(final,ensure_ascii=False))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
