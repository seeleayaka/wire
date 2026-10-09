"""Fresh source controls for proposed outgoing-component contradiction semantics."""
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from bundle_runtime_pins import source_pins
from run_review import read,save,verified_run
from run_prompt_contrast import verify
from socket_native_extent import socket_extent,POLICY

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_socket_extent_source_controls_20261006'
    protocol=read(evidence/'protocol.json');prep=read(evidence/'preparation_report.json')
    inference=read(evidence/'inference_report.json')
    if inference['status']!='complete' or inference['protocol_sha256']!=sha256(evidence/'protocol.json'):raise ValueError('fresh source inference not complete')
    verify(protocol['pins'])
    scope=read(protocol['confirmed_scope_path']);socket=next(a for a in scope['anchors'] if a['kind']=='wire_entry_socket')
    rows=[];native_count=0
    for row in prep['cases']:
        case=row.get('crop_context');pose=next((p for p in row['anchors'] if p['id']==socket['id']),None)
        audits=[];outgoing=[]
        if row['sam_inference_requested']:
            for recipe in protocol['recipes']:
                run=evidence/row['id']/recipe;manifest=read(run/'run_manifest.json')
                verified=verified_run(run,manifest['image_binding']['image_path']);native_count+=len(verified['paths'])
                if recipe!='cable_plus_reference_anatomy_box':continue
                for i,path in enumerate(verified['paths']):
                    with Image.open(path) as im:raw=np.asarray(im.convert('L'))
                    components=socket_extent(raw,pose['inspection_to_reference_local'],socket['bbox_xyxy'],tuple(case['crop_box_xyxy'][:2]))
                    audits.append({'record_id':f'mask_{i+1:03d}','score':verified['scores'][i],'components':components})
                    outgoing += [(i+1,c['component_index']) for c in components if verified['scores'][i]>=.75 and c['touches_socket_and_reaches_outgoing_context']]
        annotation=row['source_control_annotation']
        expected='mating_body_visible' if annotation is None or annotation['visual_label']==1 else 'socket_contacts_exposed'
        qualified=bool(pose is not None and pose['localization_proposal_supported'] and all(pose['gates'].values()))
        passed=bool(row['sam_inference_requested'] and qualified and row['phenotype']==expected and bool(outgoing)==(expected=='mating_body_visible'))
        rows.append({'id':row['id'],'source_annotation':annotation,'phenotype':row['phenotype'],'expected_source_phenotype':expected,
            'native_outgoing_component_at_socket':bool(outgoing),'outgoing_components':outgoing,
            'component_audit':audits,'control_passed':passed})
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    verify(protocol['pins'])
    output=ROOT/'artifacts/mendeley_socket_extent_source_gate_20261006';output.mkdir(exist_ok=False)
    report={'status':'complete','source_gate_passed':all(r['control_passed'] for r in rows),'cases':rows,
        'source_controls':4,'reference_controls':1,'native_masks_verified':native_count,
        'source_annotations_are_assistant_qualitative':True,'not_independent_accuracy':True,
        'electrical_correctness':'not_assessed','deployed':False,'policy':POLICY,
        'pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('socket_native_extent.py'),
            evidence/'protocol.json',evidence/'preparation_report.json',evidence/'inference_report.json']}}
    save(output/'report.json',report)
    print({'source_gate_passed':report['source_gate_passed'],'cases':{r['id']:r['control_passed'] for r in rows}})


if __name__=='__main__':main()
