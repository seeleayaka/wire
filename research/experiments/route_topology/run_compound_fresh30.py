"""Fresh original and global/local pose checks for the frozen contact cue."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from heldout_anchor_pose import localize
from run_positive_contact_gate import bright_contacts
from component_cues import color_descriptor, semantic_descriptor, prediction
from socket_appearance import descriptor
from socket_phenotype import predict
from socket_phenotype_agreement import feature_view, agreement
import torch

ROOT = Path(__file__).resolve().parents[2]

def main():
    out = ROOT/'artifacts/mendeley_compound_fresh30_20261006'
    out.mkdir(exist_ok=False)
    before = source_pins()
    scope_path = ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    prep_path = ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json'
    report_path = ROOT/'artifacts/mendeley_positive_contact_source_20261006/report.json'
    mask_path = report_path.parent/'fit_contact_support.png'
    pins = {str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('run_positive_contact_gate.py'),scope_path,prep_path,report_path,mask_path]}
    source_report=json.loads(report_path.read_text(encoding='utf-8'))
    if not source_report['strict_net_source_gain']: raise ValueError('source gate failed')
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    reference_case=json.loads(prep_path.read_text(encoding='utf-8'))['cases'][0]
    inventory=json.loads((ROOT/'artifacts/mendeley_visible_fan_scope_20261005/protocol.json').read_text(encoding='utf-8'))
    folder=Path(inventory['original_sources']['inspection']['path']).parent
    paths=sorted(folder.glob('*.JPG'))
    if len(paths)!=30:raise ValueError('expected30 originals')
    sources=[reference_case]+[dict(id=f'case_{j:02d}',original_source=dict(path=str(p),image_sha256=sha256(p)),source_control_annotation=None) for j,p in enumerate(paths,1)]
    pins.update({r['original_source']['path']:r['original_source']['image_sha256'] for r in sources})
    support=np.asarray(Image.open(mask_path))>0
    threshold=source_report['fit_threshold']
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,mainline_pins=before,
        rule='fixed FIT support and threshold; fresh originals/global and local CPU pose',
        no_threshold_changes=True,cue_only_not_full_classifier_or_topology=True),indent=2),encoding='utf-8')
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    torch.set_num_threads(8)
    repo=Path('E:/PythonProject10/models/dinov2')
    checkpoint=repo/'weights/dinov2_vits14_pretrain.pth'
    pins[str(checkpoint)]=sha256(checkpoint)
    if pins[str(checkpoint)]!='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9':raise ValueError('encoder drift')
    def head(path):
        pins[str(path)]=sha256(path)
        raw=json.loads(path.read_text(encoding='utf-8'))
        m={k:np.asarray(raw[k]) if k in ['center','scale','weights'] else raw[k] for k in ['center','scale','weights','bias']}
        return m,{int(k):v for k,v in raw['calibration'].items()},raw
    cm,cc,_=head(ROOT/'artifacts/mendeley_component_color_source_20261006/model.json')
    sm,sc,sraw=head(ROOT/'artifacts/mendeley_component_semantic_source_v2_20261006/model.json')
    bp=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005/model.json'
    pins[str(bp)]=sha256(bp)
    oldraw=json.loads(bp.read_text(encoding='utf-8'))
    oldmodels={v:({k:np.asarray(raw[k]) if k in ['center','scale','weights'] else raw[k] for k in ['center','scale','weights','bias']},{int(k):values for k,values in raw['calibration'].items()}) for v,raw in oldraw.items()}
    encoder=torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=False)
    encoder.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);encoder.eval()
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    (out/'model_pins.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
    rows=[]
    for r in sources:
        origin=r['original_source']
        if sha256(origin['path'])!=origin['image_sha256']:raise ValueError('original drift')
        rgb=np.asarray(Image.open(origin['path']).convert('RGB'))
        if r['id']=='reference':pose=dict(localization_proposal_supported=True,inspection_to_reference_local=np.eye(3).tolist())
        else:
            _,reg=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            if not reg['alignment_quality']['reliable']:raise ValueError('global pose unsupported')
            pose=localize(reference,rgb,anchor,np.array(reg['source_to_reference_homography']))
        if not pose['localization_proposal_supported']:raise ValueError('local pose unsupported')
        matrix=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(pose['inspection_to_reference_local'])
        valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),matrix,(100,50),flags=cv2.INTER_NEAREST)
        if not valid.all():raise ValueError('missing pixels')
        patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
        Image.fromarray(patch).save(out/(r['id']+'_fresh_socket.png'))
        score=float(bright_contacts(patch)[support].mean())
        f=descriptor(patch)
        old=agreement({v:predict(m,c,feature_view(f,v)) for v,(m,c) in oldmodels.items()})['visual_label_candidate']
        color=prediction(cm,color_descriptor(patch),cc)['visual_label_candidate']
        resized=np.asarray(Image.fromarray(patch).resize((224,112),Image.Resampling.BILINEAR)).copy()
        tensor=(torch.from_numpy(resized.transpose(2,0,1)).float()/255-mean)/std
        with torch.inference_mode():tokens=encoder.forward_features(tensor.unsqueeze(0))['x_norm_patchtokens'].squeeze(0).cpu().numpy()
        tokens=tokens/np.maximum(np.linalg.norm(tokens,axis=1,keepdims=True),1e-8)
        sf,_=semantic_descriptor(tokens,np.asarray(sraw['centers']))
        semantic=prediction(sm,sf,sc)['visual_label_candidate']
        candidate=old if old is not None else (0 if color==0 and semantic==0 and score>threshold else None)
        annotation=r['source_control_annotation']
        label=1 if r['id']=='reference' else None
        rows.append(dict(id=r['id'],source=origin,pose=pose,visual_label=label,
            contact_score=score,positive_contact_cue=score>threshold,old_candidate=old,color_candidate=color,semantic_candidate=semantic,compound_candidate=candidate))
    if before!=source_pins() or any(sha256(p)!=d for p,d in pins.items()):raise ValueError('E/code drift')
    false_cues=[r['id'] for r in rows if r['visual_label']==1 and r['positive_contact_cue']]
    missed=[r['id'] for r in rows if r['visual_label']==0 and not r['positive_contact_cue']]
    report=dict(status='complete',cases=rows,visible_false_cues=false_cues,exposed_missed=missed,
        fixed_source_cue_gate='not_evaluated_no_inspection_GT',mainline_unchanged=True,deployed=False,
        new_confirmed_connections=0,reference_gate=rows[0]['compound_candidate']==1,inspection_GT_read=False,all30_originals_and_poses_fresh=True,development_preview_not_field_accuracy=True,phenotype_counts={str(k):sum(r['compound_candidate']==k for r in rows[1:]) for k in [0,1,None]},old_phenotype_counts={str(k):sum(r['old_candidate']==k for r in rows[1:]) for k in [0,1,None]},not_topology_or_SAM_acceptance=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}));print(json.dumps([{k:r[k] for k in ['id','visual_label','old_candidate','color_candidate','semantic_candidate','compound_candidate']} for r in rows]))

if __name__=='__main__': main()




