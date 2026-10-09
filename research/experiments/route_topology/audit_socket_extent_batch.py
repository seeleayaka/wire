"""Independent native component/socket extent arithmetic and typed delta replay."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from bundle_runtime_pins import source_pins
from core import sha256
from run_prompt_contrast import verify
from run_review import read, save, verified_run

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=ROOT/'artifacts/mendeley_socket_extent_candidate_20261006'
    report=read(folder/'report.json'); verify(report['pins'])
    evidence=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    protocol=read(evidence/'protocol.json'); preparation=read(evidence/'preparation_report.json')
    scope=read(protocol['confirmed_scope_path'])
    socket=next(a for a in scope['anchors'] if a['kind']=='wire_entry_socket')
    l,t,r,b=socket['bbox_xyxy']; unit=min(r-l,b-t)
    observed={row['id']:row['observation'] for row in report['cases']}
    observed['reference']=report['reference_observation']
    replays={}; component_count=0
    for row in preparation['cases']:
        target=observed[row['id']]; outgoing=[]
        if row['sam_inference_requested'] and target['local_anchor_proposals_supported'].get(socket['id'],False):
            pose=next(p for p in row['anchors'] if p['id']==socket['id'])
            h=np.array(pose['inspection_to_reference_local']); a,d=row['crop_context']['crop_box_xyxy'][:2]
            run=evidence/row['id']/'cable_plus_reference_anatomy_box'
            manifest=read(run/'run_manifest.json'); native=verified_run(run,manifest['image_binding']['image_path'])
            for i,path in enumerate(native['paths']):
                with Image.open(path) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
                count,labels=cv2.connectedComponents(raw,connectivity=8)
                expected=target['socket_outgoing_component_audit'][i]
                if len(expected['components'])!=count-1:raise ValueError('component inventory differs')
                for index in range(1,count):
                    ys,xs=np.nonzero(labels==index); q=h@np.array([xs+a,ys+d,np.ones(len(xs))])
                    q=q[:2]/q[2:3]; x,y=q
                    hits=int(np.count_nonzero((x>=l)&(x<=r)&(y>=t)&(y<=b)))
                    extent=max(float(max(0,l-xj,xj-r,t-yj,yj-b)) for xj,yj in zip(x,y))/unit
                    candidate=bool(hits>0 and extent>=1.0)
                    item=expected['components'][index-1]
                    if item['pixel_count']!=len(xs) or item['socket_pixel_support']!=hits or abs(item['outside_extent_port_short_sides']-extent)>1e-10:
                        raise ValueError('independent socket extent arithmetic differs')
                    if item['touches_socket_and_reaches_outgoing_context']!=candidate:raise ValueError('component qualification differs')
                    if candidate and native['scores'][i]>=.75:outgoing.append((i+1,index))
                    component_count+=1
        if bool(outgoing)!=target['native_outgoing_component_at_socket']:raise ValueError('independent outgoing flag differs')
        replays[row['id']]=bool(outgoing)
    changes=[]
    for result in report['cases']:
        base=result['baseline_comparison']; expected=base['decision']
        if base.get('reason')=='exposed_socket_and_high_score_mask_conflict' and not replays[result['id']]:
            expected='visible_socket_attachment_change_supported'; changes.append(result['id'])
        if result['comparison']['decision']!=expected:raise ValueError('independent typed delta differs')
        if result['comparison']['electrical_disconnections_confirmed']!=0:raise ValueError('electrical claim introduced')
    if changes!=report['changed_cases'] or source_pins()!=protocol['mainline_pins']:raise ValueError('change inventory or E snapshot differs')
    verify(report['pins'])
    output=ROOT/'artifacts/mendeley_socket_extent_candidate_audit_20261006'; output.mkdir(exist_ok=False)
    save(output/'report.json',{'status':'PASS','independent_components_replayed':component_count,
        'changed_cases_independently_verified':changes,'old_definite_decisions_preserved':True,
        'source_group_gate':'not_run','E_deployed':False,'not_blind_or_field_accuracy':True,
        'pins':{str(p):sha256(p) for p in [Path(__file__),folder/'report.json',evidence/'protocol.json']}})
    print('PASS: independent native arithmetic and30-case typed deltas; source gate still pending')


if __name__=='__main__':main()
