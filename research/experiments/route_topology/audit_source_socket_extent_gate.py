"""Separate component/socket arithmetic; verify source controls, not electrical GT."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from bundle_runtime_pins import source_pins
from run_review import read,save,verified_run
from run_prompt_contrast import verify

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_socket_extent_source_controls_20261006'
    path=ROOT/'artifacts/mendeley_socket_extent_source_gate_20261006/report.json'
    report=read(path);protocol=read(evidence/'protocol.json');prep=read(evidence/'preparation_report.json')
    verify(protocol['pins']);verify(report['pins'])
    scope=read(protocol['confirmed_scope_path']);socket=next(a for a in scope['anchors'] if a['kind']=='wire_entry_socket')
    l,t,r,b=socket['bbox_xyxy'];unit=min(r-l,b-t);passed=[];components_checked=0
    for row,expected in zip(prep['cases'],report['cases']):
        if row['id']!=expected['id']:raise ValueError('source roster differs')
        found=False
        if row['sam_inference_requested']:
            pose=next(p for p in row['anchors'] if p['id']==socket['id']);h=np.array(pose['inspection_to_reference_local'])
            a,d=row['crop_context']['crop_box_xyxy'][:2];run=evidence/row['id']/'cable_plus_reference_anatomy_box'
            manifest=read(run/'run_manifest.json');verified=verified_run(run,manifest['image_binding']['image_path'])
            for i,maskpath in enumerate(verified['paths']):
                with Image.open(maskpath) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
                n,labels=cv2.connectedComponents(raw,connectivity=8)
                if len(expected['component_audit'][i]['components'])!=n-1:raise ValueError('native components differ')
                for part in range(1,n):
                    ys,xs=np.nonzero(labels==part);p=h@np.array([xs+a,ys+d,np.ones(len(xs))]);x,y=p[:2]/p[2:3]
                    hit=int(np.count_nonzero((x>=l)&(x<=r)&(y>=t)&(y<=b)))
                    extent=max(max(0,l-xx,xx-r,t-yy,yy-b) for xx,yy in zip(x,y))/unit
                    candidate=bool(hit>0 and extent>=1.0)
                    old=expected['component_audit'][i]['components'][part-1]
                    if hit!=old['socket_pixel_support'] or abs(extent-old['outside_extent_port_short_sides'])>1e-10 or candidate!=old['touches_socket_and_reaches_outgoing_context']:
                        raise ValueError('independent source component arithmetic differs')
                    found=found or (candidate and verified['scores'][i]>=.75);components_checked+=1
        if found!=expected['native_outgoing_component_at_socket']:raise ValueError('source outgoing evidence differs')
        source=expected['source_annotation'];state='mating_body_visible' if source is None or source['visual_label']==1 else 'socket_contacts_exposed'
        if expected['expected_source_phenotype']!=state:raise ValueError('source annotation changed')
        pose=next((p for p in row['anchors'] if p['id']==socket['id']),None)
        qualified=pose is not None and pose['localization_proposal_supported'] and all(pose['gates'].values())
        gate=bool(row['sam_inference_requested'] and qualified and row['phenotype']==state and found==(state=='mating_body_visible'))
        if gate!=expected['control_passed']:raise ValueError('source gate differs')
        passed.append(gate)
    if all(passed)!=report['source_gate_passed'] or source_pins()!=protocol['mainline_pins']:raise ValueError('aggregate source/E differs')
    output=ROOT/'artifacts/mendeley_socket_extent_source_gate_audit_20261006';output.mkdir(exist_ok=False)
    save(output/'report.json',{'status':'PASS','source_gate_passed':all(passed),'native_components_replayed':components_checked,
        'not_human_GT_or_independent_accuracy':True,'deployed':False,
        'pins':{str(p):sha256(p) for p in [Path(__file__),path,evidence/'protocol.json']}})
    print('PASS independent source replay; source gate='+str(all(passed)))


if __name__=='__main__':main()
