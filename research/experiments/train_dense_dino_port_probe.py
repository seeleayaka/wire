"""Finite local dense-head training; frozen encoder and source-disjoint crops.

Use only the audited existing656/576 rectangle crops. Fixed8 head epochs,last
checkpoint only. Crop metrics are feasibility, not full-image field accuracy.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
BASE=ROOT/'artifacts/dense_dino_port_probe_20261003'
DATA=ROOT/'artifacts/port_training_multiscale_20261002/dataset'
os.environ.update(HF_HUB_OFFLINE='1',YOLO_OFFLINE='True')
from inspection_agent.optional_port_crop_review import sha,read_image
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_multiscale_acceptance import metric
from dense_port_probe import (GRID,INPUT,EPOCHS,LR,CROP_SCORE,MIN_CROP_PRECISION,MIN_CROP_RECALL,
    DensePortHead,frozen_features,boxes_from_polygon_labels,encode_targets,loss,decode)


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def encoder_digest(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True);args=parser.parse_args()
    out=BASE/args.mode
    if out.exists():raise FileExistsError('Preserve any completed/failed/running experiment')
    manifestpath=DATA/'dataset_manifest.json';auditpath=ROOT/'artifacts/port_training_multiscale_20261002/audit_v2/report.json'
    manifest=load(manifestpath);audit=load(auditpath)
    assert manifest['status']=='complete' and audit['status']=='complete'
    records={split:[r for r in manifest['records'] if r['split']==split] for split in ('train','inner_val')}
    # Dataset uses the YAML val name but manifest source split is verified below.
    if not records['inner_val']:records['inner_val']=[r for r in manifest['records'] if r['split']=='val']
    assert len(records['train'])==656 and len(records['inner_val'])==576
    assert {r['source_image'] for r in records['train']}.isdisjoint(r['source_image'] for r in records['inner_val'])
    assert len({r['source_image'] for r in records['train']})==192
    assert len({r['source_image'] for r in records['inner_val']})==48
    if args.mode=='smoke':
        records['train']=([r for r in records['train'] if r['label_count']>0][:2]+
                          [r for r in records['train'] if r['label_count']==0][:2])
        records['inner_val']=[]
    else:
        smoke=load(BASE/'smoke/report.json')
        assert smoke['status']=='complete' and smoke['frozen_encoder_unchanged'] and smoke['nonzero_gradient_steps']>0
    weight=REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth'
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('dense_port_probe.py'),manifestpath,auditpath,weight,
          REPO/'prototype/dino_feature_diff.py',REPO/'models/dinov2/dinov2/models/vision_transformer.py')}
    for rows in records.values():
        for r in rows:
            for key,expected in (('image','image_sha256'),('label','label_sha256')):
                path=DATA/r[key];digest=sha(path);assert digest==r[expected];pins[str(path)]=digest
    frozen=resolution_runtime_fingerprint(REPO);out.mkdir(parents=True)
    save(out/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,mode=args.mode,encoder_frozen=True,
        spatial_head='384->64 shared3x3 convolution,2 sigmoid center heatmaps,2x4 center-offset/log-size readout',
        image_input=INPUT,patch_grid=GRID,epochs=1 if args.mode=='smoke' else EPOCHS,lr=LR,
        optimizer='AdamW weight_decay.01,batch8,gradient_norm_clip5',seed=20261003,
        class_ids=dict(unplugged_plug=0,unplugged_jack=1),rectangle_box_labels_not_instance_masks=True,
        no_yolo_proposals_or_ancestor_teacher_gate=True,no_filename_or_image_coordinates_as_classifier_channels=True,
        source_disjoint_train192_inner48=True,no_outer_validation_in_training=True,shared_encoder_not_updated=True,
        fixed_last_checkpoint_only=True,crop_gate=dict(score=CROP_SCORE,min_precision=MIN_CROP_PRECISION,min_recall=MIN_CROP_RECALL),
        crop_metrics_not_source_accuracy=True,validation_reused=True,no_automatic_deployment=True,field_accuracy=False,
        sources=['https://github.com/facebookresearch/dinov2','https://arxiv.org/abs/1904.07850','https://arxiv.org/abs/1904.01355']))
    import torch
    import dino_feature_diff as dino
    torch.set_num_threads(2);torch.manual_seed(20261003);random.seed(20261003)
    encoder=dino._model();encoder.requires_grad_(False);encoder.eval();torch.set_num_threads(2)
    encoder_before=encoder_digest(encoder);started=time.monotonic();cache={};cache_pins={};collisions=0
    def progress(**kw):save(out/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    def cache_split(split):
        nonlocal collisions
        folder=out/'features'/split;folder.mkdir(parents=True);items=[];rows=records[split]
        for start in range(0,len(rows),4):
            batch=rows[start:start+4];images=[read_image(DATA/r['image']) for r in batch]
            features=frozen_features(encoder,images)
            for offset,(row,image) in enumerate(zip(batch,images)):
                h,w=image.shape[:2];text=(DATA/row['label']).read_text(encoding='utf-8')
                boxes=boxes_from_polygon_labels(text,w,h);targets=encode_targets(boxes,w,h);collisions+=targets['collisions']
                assert len(boxes)==row['label_count']
                path=folder/(str(start+offset).zfill(4)+'.pt')
                torch.save(dict(features=features[offset].clone(),targets={k:targets[k] for k in ('heat','reg','mask')},
                    shape=[h,w],boxes=boxes,image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],source_image=row['source_image']),path)
                cache_pins[str(path)]=sha(path);items.append(path)
            progress(phase='frozen_features',split=split,completed=min(start+4,len(rows)),total=len(rows))
        cache[split]=items
    try:
        cache_split('train')
        head=DensePortHead();optimizer=torch.optim.AdamW(head.parameters(),lr=LR,weight_decay=.01)
        initial={k:v.detach().clone() for k,v in head.state_dict().items()};epochs=1 if args.mode=='smoke' else EPOCHS
        history=[];nonzero=0;finite=True
        for epoch in range(epochs):
            head.train();order=list(range(len(cache['train'])));random.Random(20261003+epoch).shuffle(order);totals=[]
            for start in range(0,len(order),8):
                items=[torch.load(cache['train'][i],map_location='cpu',weights_only=True) for i in order[start:start+8]]
                features=torch.stack([i['features'] for i in items]);targets={k:torch.stack([i['targets'][k] for i in items]) for k in ('heat','reg','mask')}
                optimizer.zero_grad(set_to_none=True);value,parts=loss(head(features),targets)
                assert torch.isfinite(value);value.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.);assert torch.isfinite(norm)
                nonzero+=int(float(norm)>0);optimizer.step();totals.append(float(value.detach()))
                if start%80==0:progress(phase='small_head_training',epoch=epoch+1,epochs=epochs,completed=min(start+8,len(order)),total=len(order),loss=totals[-1])
            history.append(dict(epoch=epoch+1,mean_loss=sum(totals)/len(totals)));save(out/'history.json',history)
        head.eval();changed=sum(not torch.equal(v,head.state_dict()[k]) for k,v in initial.items())
        assert changed>0 and nonzero>0
        checkpoint=out/'last_head.pt'
        torch.save(dict(state_dict=head.state_dict(),input_size=INPUT,grid=GRID,encoder_sha256=sha(weight),epochs=epochs),checkpoint)
        scores=None;qualifies=None
        if args.mode=='full':
            cache_split('inner_val');metrics=dict(tp=0,unmatched=0,fn=0,predictions=0,targets=0);evaluated=[]
            with torch.inference_mode():
                for start in range(0,len(cache['inner_val']),8):
                    items=[torch.load(p,map_location='cpu',weights_only=True) for p in cache['inner_val'][start:start+8]]
                    output=head(torch.stack([i['features'] for i in items]));predictions=decode(output,[i['shape'] for i in items],score=CROP_SCORE,max_boxes=10)
                    for offset,(item,selected) in enumerate(zip(items,predictions)):
                        value=metric(selected,item['boxes'])
                        for k,v in value.items():metrics[k]+=v
                        evaluated.append(dict(crop_index=start+offset,source_image=item['source_image'],metrics=value,predictions=selected))
                    progress(phase='inner_crop_detector_feasibility',completed=min(start+8,len(cache['inner_val'])),total=len(cache['inner_val']))
            precision=metrics['tp']/max(1,metrics['tp']+metrics['unmatched']);recall=metrics['tp']/max(1,metrics['targets'])
            scores=dict(metrics=metrics,precision=precision,recall=recall)
            qualifies=precision>=MIN_CROP_PRECISION and recall>=MIN_CROP_RECALL
            save(out/'inner_crop_evaluation.json',dict(score=CROP_SCORE,summary=scores,cases=evaluated,
                 repeated_boxes_across_crops=True,source_accuracy=False))
        assert encoder_digest(encoder)==encoder_before and not any(p.requires_grad for p in encoder.parameters())
        assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
        assert {p:sha(Path(p)) for p in cache_pins}==cache_pins
        result=dict(status='complete',mode=args.mode,completed_epochs=epochs,head_parameters_changed=changed,
            nonzero_gradient_steps=nonzero,finite_gradients=finite,frozen_encoder_unchanged=True,
            checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)),encoder_sha256=sha(weight),
            crop_feasibility=scores,qualifies_crop_feasibility=qualifies,center_cell_collisions=collisions,
            runtime_fingerprint=frozen,cache_pins=cache_pins,seconds=round(time.monotonic()-started,2),
            no_source_accuracy_yet=True,production_changed=False,field_accuracy=False)
        save(out/'report.json',result);save(out/'progress.json',dict(status='complete',mode=args.mode,
             qualifies_crop_feasibility=qualifies,seconds=result['seconds']));print(json.dumps({k:v for k,v in result.items() if k not in ('cache_pins','runtime_fingerprint')}),flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise


if __name__=='__main__':main()
