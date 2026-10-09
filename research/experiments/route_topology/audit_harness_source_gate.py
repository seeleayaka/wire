"""Independent original native pixel-set and typed-decision source replay."""
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from run_review import read, save, verified_run
from run_prompt_contrast import verify
from core import sha256
from bundle_runtime_pins import source_pins
from replay_harness_reference_trial import parts

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_harness_source_controls_20261006'
    report_path=ROOT/'artifacts/mendeley_harness_source_gate_20261006/report.json'
    report,p=read(report_path),read(evidence/'protocol.json')
    verify(p['pins']);verify(report['pins'])
    scope=read(p['confirmed_scope_path'])
    prep=read(evidence/'preparation_report.json')['cases']
    if [r['id'] for r in prep]!=[r['id'] for r in report['cases']] or len(prep)!=5:
        raise ValueError('fixed source control inventory changed')
    observations=[];checked=0
    for source,row in zip(prep,report['cases']):
        qualified={a['id']:bool(a['localization_proposal_supported'] and a['gates'] and all(a['gates'].values())) for a in source['anchors']}
        eligible=0;socket_touch=False
        if source['sam_inference_requested']:
            run=evidence/source['id']/'wire_harness_plus_reference_anatomy_box'
            manifest=read(run/'run_manifest.json')
            native=verified_run(run,manifest['image_binding']['image_path'])
            masks=row['observation']['mask_audit']
            if len(native['paths'])!=len(masks):raise ValueError('mask inventory changed')
            for path,score,mask in zip(native['paths'],native['scores'],masks):
                if score!=mask['score']:raise ValueError('score changed')
                with Image.open(path) as im:active=np.asarray(im.convert('L'))>0
                height,width=active.shape
                boundary=any(y<=1 or x<=1 or y>=height-2 or x>=width-2 for y,x in zip(*np.nonzero(active)))
                actual=[]
                for component in parts(active):
                    points=np.array([[x+source['crop_context']['crop_box_xyxy'][0],y+source['crop_context']['crop_box_xyxy'][1],1] for y,x in component])
                    hits={}
                    for anchor in scope['anchors']:
                        hits[anchor['id']]=0
                        if not qualified.get(anchor['id'],False):continue
                        pose=next(a for a in source['anchors'] if a['id']==anchor['id'])
                        mapped=(np.array(pose['inspection_to_reference_local'])@points.T).T
                        mapped=mapped[:,:2]/mapped[:,2:]
                        l,t,r,b=anchor['bbox_xyxy']
                        hits[anchor['id']]=sum(l<=x<=r and t<=y<=b for x,y in mapped)
                    good=bool(score>=.75 and not boundary and all(qualified.get(a['id'],False) for a in scope['anchors']) and all(hits.values()))
                    eligible+=int(good)
                    socket_touch=socket_touch or bool(score>=.75 and hits['FAN_CPU']>0)
                    actual.append((len(component),tuple(sorted(hits.items())),good));checked+=1
                expected=[(c['pixel_count'],tuple(sorted(c['anchor_pixel_support'].items())),c['eligible_visible_bundle_component']) for c in mask['components']]
                if Counter(actual)!=Counter(expected):raise ValueError('independent native geometry differs')
        observation=row['observation']
        if eligible!=len(observation['eligible_native_components']) or socket_touch!=observation['high_score_mask_touches_socket']:
            raise ValueError('native evidence aggregate differs')
        observations.append((source['phenotype'],qualified,source['sam_inference_requested'],eligible,socket_touch))
    state,q,verified,eligible,touch=observations[0]
    reference_ready=state=='mating_body_visible' and verified and eligible==1 and all(q.values())
    passes=[]
    for row,(state,q,verified,eligible,touch) in zip(report['cases'],observations):
        decision='insufficient_evidence'
        if reference_ready and verified and q.get('FAN_CPU',False):
            if state=='socket_contacts_exposed' and not touch:decision='visible_socket_attachment_change_supported'
            elif state=='mating_body_visible' and all(q.values()) and eligible==1:decision='same_visible_bundle_attachment_supported'
        if decision!=row['comparison']['decision']:raise ValueError('typed comparison differs')
        passed=decision==row['expected_source_decision']
        if passed!=row['control_passed']:raise ValueError('source gate differs')
        passes.append(passed)
    if all(passes)!=report['source_gate_passed'] or source_pins()!=p['mainline_pins']:raise ValueError('source aggregate/E differs')
    output=ROOT/'artifacts/mendeley_harness_source_gate_audit_20261006';output.mkdir(exist_ok=False)
    save(output/'report.json',{'status':'PASS','source_gate_passed':all(passes),'native_components_replayed':checked,
        'not_independent_field_accuracy':True,'deployed':False,
        'pins':{str(f):sha256(f) for f in [Path(__file__),report_path,evidence/'protocol.json']}})
    print({'independent_replay':'PASS','source_gate_passed':all(passes)})


if __name__=='__main__':main()
