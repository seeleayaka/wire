"""Fresh local footprints from native files, no SAM rerun, decisions unchanged."""
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_wire_local_shape_trial import prepare,ROOT,END,APPEAR
from run_wire_appearance_trial import region_masks
from reference_wire_appearance_v3 import assess
from endpoint_neighborhood_competition import assess_competition
from run_prompt_contrast import save,digest,verify
from run_review import verified_run
from bundle_runtime_pins import source_pins

OUT=ROOT/'artifacts/endpoint_neighborhood_competition_20261008'


def main():
    start=time.monotonic();protocol,prior,_,prepared,scope,_=prepare()
    before=source_pins();assert before==protocol['mainline_pins']
    files=[Path(__file__),Path(__file__).with_name('endpoint_neighborhood_competition.py'),
           Path(__file__).with_name('test_endpoint_neighborhood_competition.py'),
           APPEAR/'report.json',APPEAR/'protocol.json',END/'report.json',END/'protocol.json']
    pins={str(f):digest(f) for f in files}
    OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,fresh_SAM_calls=0,
        policy='exact local colored-footprint aliases only; different footprints request review, not identity rejection'))
    cases=[]
    for row in prior['cases']:
        records=[]
        if row['native_run_directory']:
            crop=row['crop_box_xyxy'];source=row['source_path'];case=prepared[row['id']]
            assert digest(source)==row['source_binding']['image_sha256']
            rgb=np.asarray(Image.open(source).convert('RGB').crop(crop))
            assert np.array_equal(rgb,np.asarray(Image.open(case['crop_context']['source']['path']).convert('RGB')))
            inv=verified_run(row['native_run_directory'],case['crop_context']['source']['path'])
            regions,scales=region_masks(rgb.shape[:2],crop,scope,case['anchors'])
            for path,score in zip(inv['paths'],inv['scores']):
                raw=np.asarray(Image.open(path).convert('L'))>0;endpoints={}
                for identity,region in regions.items():
                    color,pixels=assess(rgb,raw&region,protocol['reference_profiles'][identity],scales[identity])
                    endpoints[identity]=dict(appearance_state=color['state'],matched_pixels=pixels)
                records.append(dict(record_id=path.stem,score=float(score),endpoints=endpoints))
        result=assess_competition(records,[a['id'] for a in scope['anchors']])
        cases.append(dict(id=row['id'],old_decision=row['decision'],old_decision_unchanged=True,competition=result))
    verify(pins);assert source_pins()==before
    report=dict(status='complete',cases=cases,total_cases=len(cases),
        neighborhood_competition_case_ids=[r['id'] for r in cases if r['competition']['review_needed']],
        old19_candidate_competition_case_ids=[r['id'] for r in cases if r['old_decision']=='reference_endpoint_pair_candidate' and r['competition']['review_needed']],
        old_candidate_support_preserved=19,original_decisions_unchanged=True,
        physical_identity_confirmed=0,electrical_connections_confirmed=0,
        fresh_SAM_calls=0,not_deployed=True,not_field_accuracy=True,seconds=time.monotonic()-start)
    save(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':main()
