"""Fixed16-epoch training-only augmentation of verified epoch8 head."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'E:/PythonProject10')
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_multiscale_acceptance import metric
from dense_pixel_port_probe import DensePixelPortHead,loss,decode,INPUT,GRID,CROP_SCORE,MIN_CROP_PRECISION,MIN_CROP_RECALL
from dense_pixel_geometric_augmentation import transform_item
OLD=ROOT/'artifacts/dense_pixel_port_probe_20261003/full'
BASE=ROOT/'artifacts/dense_pixel_augmented_20261003'
REPO=Path('E:/PythonProject10')
ADDITIONAL_EPOCHS=16
TOTAL_EPOCHS=24


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True);args=parser.parse_args()
    out=BASE/args.mode
    if out.exists():raise FileExistsError('Preserve prior experiment')
    before=load(OLD/'report.json');old_protocol=load(OLD/'protocol.json')
    assert before['status']=='complete' and before['completed_epochs']==8 and before['frozen_encoder_unchanged']
    if args.mode=='full':assert load(BASE/'smoke/report.json')['status']=='complete'
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==before['runtime_fingerprint']
    pins=dict(old_protocol['pins'],**before['cache_pins'])
    for path in (OLD/'report.json',OLD/'protocol.json',Path(before['checkpoint']['path']),Path(__file__),
                 Path(__file__).with_name('dense_pixel_geometric_augmentation.py'),Path(__file__).with_name('dense_pixel_port_probe.py'),
                 ROOT/'artifacts/dense_pixel_augmented_preregistration_20261003/PLAN.md'):
        pins[str(path)]=sha(path)
    assert sha(Path(before['checkpoint']['path']))==before['checkpoint']['sha256']
    assert {p:sha(Path(p)) for p in pins}==pins
    dataset=ROOT/'artifacts/port_training_multiscale_20261002/dataset/dataset_manifest.json'
    rows=load(dataset)['records'];tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='val']
    assert len(tr)==656 and len(va)==576
    assert len({r['source_image'] for r in tr})==192 and len({r['source_image'] for r in va})==48
    assert {r['source_image'] for r in tr}.isdisjoint(r['source_image'] for r in va)
    out.mkdir(parents=True);started=time.monotonic()
    save(out/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,base_epoch8_checkpoint=before['checkpoint'],
        source_groups_disjoint=True,fixed_additional_epochs=1 if args.mode=='smoke' else ADDITIONAL_EPOCHS,
        fixed_total_epochs=9 if args.mode=='smoke' else TOTAL_EPOCHS,train_only_uniform_dihedral=True,
        cached_feature_augmentation_not_fresh_DINO_equivariance=True,validation_unaugmented=True,
        optimizer='AdamW .0005 cosine to .00005,decay.01,batch8,clip5',fixed_last_checkpoint=True,
        same_scores_iou_and_budget=True,no_early_stopping_or_score_search=True,field_accuracy=False))
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261003);random.seed(20261003)
    head=DensePixelPortHead();head.load_state_dict(torch.load(before['checkpoint']['path'],map_location='cpu',weights_only=True)['state_dict'])
    initial={k:v.detach().clone() for k,v in head.state_dict().items()}
    epochs=1 if args.mode=='smoke' else ADDITIONAL_EPOCHS
    paths=[OLD/'features/train'/f'{i:04d}.pt' for i in range(656 if args.mode=='full' else 8)]
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0005,weight_decay=.01)
    history=[];nonzero=0;counts={str((r,m)):0 for r in range(4) for m in (False,True)}
    def progress(**kw):save(out/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    try:
        for epoch in range(epochs):
            lr=.00005+(.0005-.00005)*.5*(1+math.cos(math.pi*epoch/max(1,epochs-1)))
            for group in optimizer.param_groups:group['lr']=lr
            head.train();order=list(range(len(paths)));rng=random.Random(20261003+epoch);rng.shuffle(order);values=[]
            for start in range(0,len(order),8):
                batch=[]
                for i in order[start:start+8]:
                    item=torch.load(paths[i],map_location='cpu',weights_only=True)
                    assert item['image_sha256']==tr[i]['image_sha256'] and item['label_sha256']==tr[i]['label_sha256'] and item['source_image']==tr[i]['source_image']
                    rotation,mirror=rng.randrange(4),bool(rng.randrange(2));counts[str((rotation,mirror))]+=1
                    batch.append(transform_item(item,rotation,mirror))
                features={k:torch.stack([v['features'][k] for v in batch]) for k in ('semantic','rgb')}
                targets={k:torch.stack([v['targets'][k] for v in batch]) for k in ('heat','reg','mask')}
                optimizer.zero_grad(set_to_none=True);value,parts=loss(head(features),targets);assert torch.isfinite(value);value.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.);assert torch.isfinite(norm);nonzero+=int(float(norm)>0)
                optimizer.step();values.append(float(value.detach()))
                if start%80==0:progress(phase='augmented_training',epoch=epoch+1,epochs=epochs,completed=min(start+8,len(paths)),total=len(paths),loss=values[-1],loss_parts=parts)
            history.append(dict(additional_epoch=epoch+1,lr=lr,mean_loss=sum(values)/len(values)));save(out/'history.json',history)
        head.eval();changed=sum(not torch.equal(v,head.state_dict()[k]) for k,v in initial.items());assert changed and nonzero
        checkpoint=out/'last_head.pt';total=8+epochs
        torch.save(dict(state_dict=head.state_dict(),input_size=INPUT,grid=GRID,encoder_sha256=before['encoder_sha256'],epochs=total,pixel_branch=True),checkpoint)
        scores=None;qualifies=None
        if args.mode=='full':
            metrics={k:0 for k in ('tp','unmatched','fn','predictions','targets')};evaluated=[]
            with torch.inference_mode():
                for start in range(0,576,8):
                    batch=[torch.load(OLD/'features/inner_val'/f'{i:04d}.pt',map_location='cpu',weights_only=True) for i in range(start,min(start+8,576))]
                    features={k:torch.stack([v['features'][k] for v in batch]) for k in ('semantic','rgb')}
                    predictions=decode(head(features),[v['shape'] for v in batch],score=CROP_SCORE,max_boxes=10)
                    for offset,(item,selected) in enumerate(zip(batch,predictions)):
                        value=metric(selected,item['boxes'])
                        for k,v in value.items():metrics[k]+=v
                        evaluated.append(dict(crop_index=start+offset,source_image=item['source_image'],metrics=value,predictions=selected))
                    progress(phase='unaugmented_inner_crop_feasibility',completed=min(start+8,576),total=576)
            precision=metrics['tp']/max(1,metrics['tp']+metrics['unmatched']);recall=metrics['tp']/max(1,metrics['targets'])
            scores=dict(metrics=metrics,precision=precision,recall=recall);qualifies=precision>=MIN_CROP_PRECISION and recall>=MIN_CROP_RECALL
            save(out/'inner_crop_evaluation.json',dict(score=CROP_SCORE,summary=scores,cases=evaluated,repeated_boxes_across_crops=True,source_accuracy=False))
        assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
        result=dict(status='complete',completed_epochs=total,additional_epochs=epochs,head_parameters_changed=changed,
            nonzero_gradient_steps=nonzero,finite_gradients=True,augmentation_counts=counts,validation_unaugmented=True,
            frozen_encoder_unchanged=True,encoder_sha256=before['encoder_sha256'],runtime_fingerprint=frozen,
            checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)),crop_feasibility=scores,qualifies_crop_feasibility=qualifies,
            seconds=round(time.monotonic()-started,2),production_changed=False,field_accuracy=False)
        save(out/'report.json',result);save(out/'progress.json',dict(status='complete',qualifies_crop_feasibility=qualifies,seconds=result['seconds']))
        print(json.dumps({k:v for k,v in result.items() if k!='runtime_fingerprint'}),flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
