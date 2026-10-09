import json
from pathlib import Path
from socket_observation_layer import summarize_socket_observation
from run_prompt_contrast import save,digest,verify
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/socket_observation_layer_20261008'


def main():
    old=ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008'
    original=json.loads((old/'report.json').read_text(encoding='utf-8'))
    p=json.loads((old/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    before=source_pins();assert before==p['mainline_pins']
    OUT.mkdir(exist_ok=False)
    files=[old/'report.json',old/'protocol.json',Path(__file__),Path(__file__).with_name('socket_observation_layer.py')]
    pins={str(f):digest(f) for f in files};rows=[]
    for row in original['cases']:
        assert digest(row['source_path'])==row['source_binding']['image_sha256']
        result=summarize_socket_observation(row,True)
        assert {k:v for k,v in result.items() if k!='socket_observation'}==row
        rows.append(result)
    verify(pins);assert source_pins()==before
    report=dict(status='complete',cases=rows,pins=pins,mainline_pins=before,
        explicit_exposure_observation_ids=[r['id'] for r in rows if r['socket_observation']['state']=='socket_contacts_exposed_observed'],
        conflict_observation_preserved_ids=[r['id'] for r in rows if r['socket_observation']['state']=='socket_contacts_exposed_observed' and r['socket_evidence_conflict']],
        original_decisions_unchanged=True,old19_candidates_preserved=True,
        new_classifier_detections=0,new_confirmed_faults=0,new_confirmed_connections=0,
        improvement_kind='evidence_visibility_not_new_recognition_accuracy',fresh_SAM_calls=0,not_deployed_to_E=True)
    save(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','pins','mainline_pins']}))


if __name__=='__main__':main()
