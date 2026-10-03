"""Train only the RGB/fine decoder; reuse fully verified frozen DINO maps."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
sys.path.insert(0,str(REPO))
BASE=ROOT/'artifacts/dense_pixel_port_probe_20261003'
COARSE=ROOT/'artifacts/dense_dino_port_probe_20261003/full'
DATA=ROOT/'artifacts/port_training_multiscale_20261002/dataset'
from inspection_agent.optional_port_crop_review import sha,read_image
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_multiscale_acceptance import metric
from dense_pixel_port_probe import (DensePixelPortHead,GRID,INPUT,EPOCHS,LR,preprocess,boxes_from_polygon_labels,
    encode_targets,loss,decode,CROP_SCORE,MIN_CROP_PRECISION,MIN_CROP_RECALL)


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True);args=parser.parse_args()
    out=BASE/args.mode
    if out.exists():raise FileExistsError('Preserve experiment')
    coarse=load(COARSE/'report.json');assert coarse['status']=='complete' and coarse['frozen_encoder_unchanged']
    manifestpath=DATA/'dataset_manifest.json';manifest=load(manifestpath)
    groups=dict(train=[r for r in manifest['records'] if r['split']=='train'],inner_val=[r for r in manifest['records'] if r['split']=='val'])
    assert len(groups['train'])==656 and len(groups['inner_val'])==576
    assert {r['source_image'] for r in groups['train']}.isdisjoint(r['source_image'] for r in groups['inner_val'])
    if args.mode=='smoke':
        indices=([i for i,r in enumerate(groups['train']) if r['label_count']>0][:2]+
                 [i for i,r in enumerate(groups['train']) if r['label_count']==0][:2])
        groups['train']=[dict(groups['train'][i],semantic_index=i) for i in indices];groups['inner_val']=[]
    else:
        smoke=load(BASE/'smoke/report.json');assert smoke['status']=='complete' and smoke['changed_pixel_tensors']>0
        groups={s:[dict(r,semantic_index=i) for i,r in enumerate(rows)] for s,rows in groups.items()}
    encoderpath=REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth';assert sha(encoderpath)==coarse['encoder_sha256']
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('dense_pixel_port_probe.py'),
         Path(__file__).with_name('dense_port_probe.py'),COARSE/'report.json',manifestpath,encoderpath,
         ROOT/'artifacts/dense_pixel_probe_preregistration_20261003/PLAN.md')}
    for rows in groups.values():
        for r in rows:
            for k,d in (('image','image_sha256'),('label','label_sha256')):
                p=DATA/r[k];assert sha(p)==r[d];pins[str(p)]=r[d]
            semantic=COARSE/'features'/('train' if r['split']=='train' else 'inner_val')/(str(r['semantic_index']).zfill(4)+'.pt')
            assert sha(semantic)==coarse['cache_pins'][str(semantic)];pins[str(semantic)]=sha(semantic)
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==coarse['runtime_fingerprint'];out.mkdir(parents=True)
    save(out/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,semantic_backbone_frozen=True,
         verified_coarse_features_reused=True,fine_grid=GRID,pixel_branch_stride=8,head_input=INPUT,
         architecture='RGB3->16->16->16 stride2 each;DINO384->32 projection;concat48->64 context3x3;center/box56grid',
         fixed_epochs=1 if args.mode=='smoke' else EPOCHS,lr=LR,optimizer='AdamW decay.01,batch8,clip5',
         sampling_grid_audit_not_model_score_triggered=True,coarse_failure_not_relabeled=True,
         same_cutoffs_and_labels=True,no_filename_or_coordinate_classifier_channels=True,
         fixed_last_checkpoint=True,train192_inner48_source_disjoint=True,validation_reused=True,
         crop_gate=dict(score=CROP_SCORE,min_precision=MIN_CROP_PRECISION,min_recall=MIN_CROP_RECALL),
         no_yolo_proposals=True,no_automatic_deployment=True,field_accuracy=False))
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261003);random.seed(20261003)
    started=time.monotonic();cache={};cache_pins={};collisions=0
    def progress(**kw):save(out/'progress.json',dict(status='running',pid=__import__('os').getpid(),seconds=round(time.monotonic()-started,2),**kw))
    def cache_split(split):
        nonlocal collisions
        folder=out/'features'/split;folder.mkdir(parents=True);items=[]
        for index,row in enumerate(groups[split]):
            semanticpath=COARSE/'features'/split/(str(row['semantic_index']).zfill(4)+'.pt')
            old=torch.load(semanticpath,map_location='cpu',weights_only=True)
            assert old['image_sha256']==row['image_sha256'] and old['label_sha256']==row['label_sha256'] and old['source_image']==row['source_image']
            image=read_image(DATA/row['image']);h,w=image.shape[:2];assert old['shape']==[h,w]
            boxes=boxes_from_polygon_labels((DATA/row['label']).read_text(encoding='utf-8'),w,h);targets=encode_targets(boxes,w,h);collisions+=targets['collisions']
            assert len(boxes)==row['label_count'];path=folder/(str(index).zfill(4)+'.pt')
            torch.save(dict(features=dict(semantic=old['features'],rgb=preprocess(image).to(torch.float16)),
                targets={k:targets[k] for k in ('heat','reg','mask')},shape=[h,w],boxes=boxes,
                image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],source_image=row['source_image']),path)
            items.append(path);cache_pins[str(path)]=sha(path)
            if index%32==0:progress(phase='cached_semantics_plus_pixels',split=split,completed=index+1,total=len(groups[split]))
        cache[split]=items
    try:
        cache_split('train');head=DensePixelPortHead();initial={k:v.detach().clone() for k,v in head.state_dict().items()}
        optimizer=torch.optim.AdamW(head.parameters(),lr=LR,weight_decay=.01);epochs=1 if args.mode=='smoke' else EPOCHS
        history=[];nonzero=0
        for epoch in range(epochs):
            head.train();order=list(range(len(cache['train'])));random.Random(20261003+epoch).shuffle(order);values=[]
            for start in range(0,len(order),8):
                batch=[torch.load(cache['train'][i],map_location='cpu',weights_only=True) for i in order[start:start+8]]
                features={k:torch.stack([i['features'][k] for i in batch]) for k in ('semantic','rgb')}
                targets={k:torch.stack([i['targets'][k] for i in batch]) for k in ('heat','reg','mask')}
                optimizer.zero_grad(set_to_none=True);value,_=loss(head(features),targets);assert torch.isfinite(value);value.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.);assert torch.isfinite(norm);nonzero+=int(float(norm)>0)
                optimizer.step();values.append(float(value.detach()))
                if start%80==0:progress(phase='fine_head_training',epoch=epoch+1,epochs=epochs,completed=min(start+8,len(order)),total=len(order),loss=values[-1])
            history.append(dict(epoch=epoch+1,mean_loss=sum(values)/len(values)));save(out/'history.json',history)
        head.eval();changed=sum(not torch.equal(v,head.state_dict()[k]) for k,v in initial.items())
        pixel_changed=sum(k.startswith('pixel.') and not torch.equal(v,head.state_dict()[k]) for k,v in initial.items())
        assert changed>0 and pixel_changed>0 and nonzero>0
        checkpoint=out/'last_head.pt';torch.save(dict(state_dict=head.state_dict(),input_size=INPUT,grid=GRID,
             encoder_sha256=coarse['encoder_sha256'],epochs=epochs,pixel_branch=True),checkpoint)
        scores=None;qualifies=None
        if args.mode=='full':
            cache_split('inner_val');metrics=dict(tp=0,unmatched=0,fn=0,predictions=0,targets=0);evaluated=[]
            with torch.inference_mode():
                for start in range(0,len(cache['inner_val']),8):
                    batch=[torch.load(p,map_location='cpu',weights_only=True) for p in cache['inner_val'][start:start+8]]
                    features={k:torch.stack([i['features'][k] for i in batch]) for k in ('semantic','rgb')}
                    predictions=decode(head(features),[i['shape'] for i in batch],score=CROP_SCORE,max_boxes=10)
                    for offset,(item,selected) in enumerate(zip(batch,predictions)):
                        value=metric(selected,item['boxes'])
                        for k,v in value.items():metrics[k]+=v
                        evaluated.append(dict(crop_index=start+offset,source_image=item['source_image'],metrics=value,predictions=selected))
                    progress(phase='fine_inner_crop_feasibility',completed=min(start+8,len(cache['inner_val'])),total=len(cache['inner_val']))
            precision=metrics['tp']/max(1,metrics['tp']+metrics['unmatched']);recall=metrics['tp']/max(1,metrics['targets'])
            scores=dict(metrics=metrics,precision=precision,recall=recall);qualifies=precision>=MIN_CROP_PRECISION and recall>=MIN_CROP_RECALL
            save(out/'inner_crop_evaluation.json',dict(score=CROP_SCORE,summary=scores,cases=evaluated,repeated_boxes_across_crops=True,source_accuracy=False))
        assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
        assert {p:sha(Path(p)) for p in cache_pins}==cache_pins
        result=dict(status='complete',mode=args.mode,completed_epochs=epochs,head_parameters_changed=changed,changed_pixel_tensors=pixel_changed,
             nonzero_gradient_steps=nonzero,finite_gradients=True,frozen_encoder_unchanged=True,encoder_not_loaded_for_new_training=True,
             checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)),encoder_sha256=coarse['encoder_sha256'],
             crop_feasibility=scores,qualifies_crop_feasibility=qualifies,center_cell_collisions=collisions,
             runtime_fingerprint=frozen,cache_pins=cache_pins,seconds=round(time.monotonic()-started,2),production_changed=False,field_accuracy=False)
        save(out/'report.json',result);save(out/'progress.json',dict(status='complete',mode=args.mode,qualifies_crop_feasibility=qualifies,seconds=result['seconds']))
        print(json.dumps({k:v for k,v in result.items() if k not in ('cache_pins','runtime_fingerprint')}),flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise


if __name__=='__main__':main()
