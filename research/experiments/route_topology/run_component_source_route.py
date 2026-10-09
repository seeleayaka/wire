"""Two isolated source-only routes. Existing reviewed source poses are pinned.

Fresh originals/features, original FIT117 + CAL80 partition, leave-self-out CAL
evaluation. No inspection images/GT/filename features. Not full ComAD/CSAD.
"""
import argparse
import json
from pathlib import Path
import time
import sys
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from component_cues import POLICY,color_descriptor,color_components,semantic_descriptor,fit_head,probability,prediction,gate
from socket_phenotype_agreement import feature_view

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--route',choices=['color','semantic'],required=True);args=parser.parse_args()
    output=ROOT/f'artifacts/mendeley_component_{args.route}_source_20261006';output.mkdir(exist_ok=False)
    def save(name,data): (output/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    begun=time.monotonic()
    label_path=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    partition_path=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json'
    labels=json.loads(label_path.read_text(encoding='utf-8'));partition=json.loads(partition_path.read_text(encoding='utf-8'))
    byid={r['id']:r for r in labels['rows']}
    fit_ids=partition['fit_ids'];cal_ids=partition['calibration_ids'];selected=fit_ids+cal_ids
    if len(fit_ids)!=117 or len(cal_ids)!=80 or set(fit_ids)&set(cal_ids):raise ValueError('frozen partition changed')
    before=source_pins();approval=json.loads((ROOT/'artifacts/mendeley_reference_confirmed_20261006/approval_record.json').read_text(encoding='utf-8'))
    if before!=approval['mainline_pins']:raise ValueError('E drift')
    baseline_path=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005/model.json'
    baseline_report=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005/report.json'
    baseline_feature_path=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/report.json'
    files=[Path(__file__),Path(__file__).with_name('component_cues.py'),label_path,partition_path,baseline_path,baseline_report,baseline_feature_path]
    pins={str(p):sha256(p) for p in files}
    pins.update({byid[i]['source_path']:byid[i]['source_sha256'] for i in selected})
    protocol=dict(route=args.route,policy=POLICY,pins=pins,mainline_pins=before,fit_ids=fit_ids,calibration_ids=cal_ids,
        source_annotations=labels['label_status'],fresh_original_decode=True,existing_reviewed_source_poses_reused=True,
        source_labels_not_human_GT=True,calibration_evaluation='leave_self_out_per_class',
        prior_development_calibration_not_independent_field_test=True,
        full_official_algorithm_reproduced=False,inspection_images_or_labels_read=False,
        component_mask_is_appearance_cue_not_wire_identity=True,no_training_DINO_encoder=True)
    if args.route=='semantic':
        import torch
        torch.set_num_threads(8)
        checkpoint=Path('E:/PythonProject10/models/dinov2/weights/dinov2_vits14_pretrain.pth')
        if sha256(checkpoint)!='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9':raise ValueError('encoder drift')
        repo=Path('E:/PythonProject10/models/dinov2')
        pins[str(checkpoint)]=sha256(checkpoint)
        pins.update({str(p):sha256(p) for p in [repo/'hubconf.py',*sorted((repo/'dinov2').rglob('*.py'))]})
        protocol.update(component_backbone='DINOv2 ViT-S14, unlike official ComAD DINOv1',
                        token_grid_hw=[8,16],input_wh=[224,112],clusters=12,prototype_fit='only97 visible FIT sources',
                        crf_used=False,full_ComAD_or_CSAD_run=False)
    save('protocol.json',protocol)
    patches={};features={};tokens={}
    if args.route=='semantic':
        encoder=torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=False)
        encoder.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);encoder.eval()
        mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    for index,identity in enumerate(selected):
        row=byid[identity]
        if sha256(row['source_path'])!=row['source_sha256'] or not row['pose']['localization_proposal_supported']:raise ValueError('source/pose drift')
        rgb=np.asarray(Image.open(row['source_path']).convert('RGB'))
        matrix=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(row['pose']['inspection_to_reference_local'])
        valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),matrix,(100,50),flags=cv2.INTER_NEAREST)
        if not valid.all():raise ValueError('missing original pixels')
        patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
        if sha256(row['patch_path'])!=row['patch_sha256'] or not np.array_equal(patch,np.asarray(Image.open(row['patch_path']).convert('RGB'))):raise ValueError('reviewed pose/pixel mismatch')
        patches[identity]=patch
        Image.fromarray(patch).save(output/(identity+'_fresh_socket.png'))
        if args.route=='color':features[identity]=color_descriptor(patch)
        else:
            im=np.asarray(Image.fromarray(patch).resize((224,112),Image.Resampling.BILINEAR)).copy()
            tensor=(torch.from_numpy(im.transpose(2,0,1)).float()/255-mean)/std
            with torch.inference_mode():raw=encoder.forward_features(tensor.unsqueeze(0))['x_norm_patchtokens'].squeeze(0).cpu().numpy()
            if raw.shape!=(128,384):raise ValueError('wrong last-block DINO token grid')
            raw=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-8)
            tokens[identity]=raw
        if index%10==0:save('progress.json',dict(status='fresh_source_features',completed=index+1,total=len(selected),seconds=time.monotonic()-begun))
    prototype=None
    if args.route=='semantic':
        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_limits
        normal_tokens=np.concatenate([tokens[i] for i in fit_ids if byid[i]['visual_label']==1])
        with threadpool_limits(limits=8):
            clusters=MiniBatchKMeans(n_clusters=12,random_state=0,n_init=1,batch_size=1024,max_iter=100,reassignment_ratio=0).fit(normal_tokens)
        prototype=clusters.cluster_centers_
        for identity in selected:
            features[identity],componentmap=semantic_descriptor(tokens[identity],prototype)
            Image.fromarray(componentmap.astype(np.uint8)*20).resize((400,200),Image.Resampling.NEAREST).save(output/(identity+'_componentmap.png'))
        np.savez_compressed(output/'tokens.npz',**tokens)
    np.savez_compressed(output/'features.npz',**features)
    fitted=fit_head([features[i] for i in fit_ids],[byid[i]['visual_label'] for i in fit_ids])
    calibration={k:[dict(id=i,score=probability(fitted,features[i]) if k==0 else 1-probability(fitted,features[i])) for i in cal_ids if byid[i]['visual_label']==k] for k in [0,1]}
    evaluated=[dict(id=i,visual_label=byid[i]['visual_label'],prediction=prediction(fitted,features[i],calibration,i)) for i in cal_ids]
    # Fair baseline replay with the same leave-self-out ranks, not its optimistic
    # archived self-included calibration coverage.
    oldmodels=json.loads(baseline_path.read_text(encoding='utf-8'))
    oldfeatures={r['id']:np.array(r['descriptor']) for r in json.loads(baseline_feature_path.read_text(encoding='utf-8'))['prepared']}
    baseline_predictions={i:[] for i in cal_ids}
    for view,raw in oldmodels.items():
        model={k:np.asarray(raw[k]) if k in ['center','scale','weights'] else raw[k] for k in ['center','scale','weights','bias']}
        oldcal={k:[dict(id=i,score=probability(model,feature_view(oldfeatures[i],view)) if k==0 else 1-probability(model,feature_view(oldfeatures[i],view))) for i in cal_ids if byid[i]['visual_label']==k] for k in [0,1]}
        for i in cal_ids:baseline_predictions[i].append(prediction(model,feature_view(oldfeatures[i],view),oldcal,i)['visual_label_candidate'])
    baseline=[dict(id=i,visual_label=byid[i]['visual_label'],prediction=dict(visual_label_candidate=values[0] if values[0] is not None and len(set(values))==1 else None)) for i,values in baseline_predictions.items()]
    old_byid={r['id']:r for r in baseline}
    losses=[r['id'] for r in evaluated if old_byid[r['id']]['prediction']['visual_label_candidate']==r['visual_label'] and r['prediction']['visual_label_candidate']!=r['visual_label']]
    gains=[r['id'] for r in evaluated if old_byid[r['id']]['prediction']['visual_label_candidate'] is None and r['prediction']['visual_label_candidate']==r['visual_label']]
    result_gate=gate(evaluated)
    model_payload={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fitted.items()}
    model_payload.update(calibration=calibration,centers=prototype.tolist() if prototype is not None else None)
    save('model.json',model_payload)
    if any(sha256(p)!=d for p,d in pins.items()) or before!=source_pins():raise ValueError('code/source/E drift')
    report=dict(status='complete',route=args.route,source_gate=result_gate,calibration_results=evaluated,
        baseline_LOO_gate=gate(baseline),baseline_LOO_results=baseline,source_gains=gains,source_losses=losses,
        strict_net_source_gain=result_gate['passed'] and len(gains)>0 and not losses,
        seconds=time.monotonic()-begun,new_confirmed_connections=0,mainline_unchanged=True,deployed=False,
        inspection_images_or_labels_read=False,source_annotations_not_human_GT=True,
        semantic_training_is_only_prototypes_and_small_head_not_encoder=args.route=='semantic',
        source_calibration_reused_development_not_field_accuracy=True,
        next='fresh fixed-source controls then independent visual review' if result_gate['passed'] and gains and not losses else 'stop before inspection; preserve source failure/non-gain')
    save('report.json',report);save('progress.json',dict(status='complete',seconds=report['seconds']))
    print(json.dumps({k:report[k] for k in ['route','source_gate','baseline_LOO_gate','source_gains','source_losses','strict_net_source_gain','seconds']}))

if __name__=='__main__':main()
