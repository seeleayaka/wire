"""Ablate augmentation only; fixed16 additional epochs, last checkpoint."""
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from train_dense_pixel_augmented import ROOT,REPO,OLD,load,save,sha
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from dense_pixel_port_probe import DensePixelPortHead,loss,decode,INPUT,GRID,CROP_SCORE,MIN_CROP_PRECISION,MIN_CROP_RECALL
from audit_port_multiscale_acceptance import metric
BASE=ROOT/'artifacts/dense_pixel_plain_20261003'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True);args=parser.parse_args()
    out=BASE/args.mode
    if out.exists():raise FileExistsError('Preserve prior run')
    previous=load(OLD/'report.json');assert previous['status']=='complete' and previous['completed_epochs']==8 and previous['frozen_encoder_unchanged']
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==previous['runtime_fingerprint']
    if args.mode=='full':assert load(BASE/'smoke/report.json')['status']=='complete'
    pins=dict(load(OLD/'protocol.json')['pins'],**previous['cache_pins'])
    for p in (Path(__file__),Path(__file__).with_name('train_dense_pixel_augmented.py'),Path(__file__).with_name('dense_pixel_port_probe.py'),
              OLD/'report.json',OLD/'protocol.json',Path(previous['checkpoint']['path']),ROOT/'artifacts/dense_pixel_plain_preregistration_20261003/PLAN.md'):pins[str(p)]=sha(p)
    assert sha(Path(previous['checkpoint']['path']))==previous['checkpoint']['sha256']
    assert {p:sha(Path(p)) for p in pins}==pins
    dataset=ROOT/'artifacts/port_training_multiscale_20261002/dataset/dataset_manifest.json'
    rows=load(dataset)['records'];tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='val']
    assert len(tr)==656 and len(va)==576
    assert len({r['source_image'] for r in tr})==192 and len({r['source_image'] for r in va})==48
    assert {r['source_image'] for r in tr}.isdisjoint(r['source_image'] for r in va)
    epochs=1 if args.mode=='smoke' else 16;total=8+epochs;out.mkdir(parents=True)
    save(out/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,initial_checkpoint=previous['checkpoint'],
        fixed_additional_epochs=epochs,fixed_total_epochs=total,augmentation=False,only_difference_from_D4_continuation='plain training inputs',
        optimizer='AdamW .0005 cosine to .00005,decay.01,batch8,clip5',fixed_last_checkpoint=True,
        source_groups_disjoint=True,no_early_stopping_or_threshold_search=True,validation_reused=True,field_accuracy=False))
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261003);random.seed(20261003)
    head=DensePixelPortHead();head.load_state_dict(torch.load(previous['checkpoint']['path'],map_location='cpu',weights_only=True)['state_dict'])
    initial={k:v.detach().clone() for k,v in head.state_dict().items()};optimizer=torch.optim.AdamW(head.parameters(),lr=.0005,weight_decay=.01)
    started=time.monotonic();history=[];nonzero=0;count=656 if args.mode=='full' else 8
    def progress(**kw):save(out/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    try:
        for epoch in range(epochs):
            lr=.00005+(.0005-.00005)*.5*(1+math.cos(math.pi*epoch/max(1,epochs-1)))
            for group in optimizer.param_groups:group['lr']=lr
            head.train();order=list(range(count));random.Random(20261003+epoch).shuffle(order);values=[]
            for start in range(0,count,8):
                batch=[]
                for i in order[start:start+8]:
                    item=torch.load(OLD/'features/train'/f'{i:04d}.pt',map_location='cpu',weights_only=True)
                    assert item['image_sha256']==tr[i]['image_sha256'] and item['label_sha256']==tr[i]['label_sha256'] and item['source_image']==tr[i]['source_image']
                    batch.append(item)
                features={k:torch.stack([v['features'][k] for v in batch]) for k in ('semantic','rgb')}
                targets={k:torch.stack([v['targets'][k] for v in batch]) for k in ('heat','reg','mask')}
                optimizer.zero_grad(set_to_none=True);value,parts=loss(head(features),targets);assert torch.isfinite(value);value.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.);assert torch.isfinite(norm);nonzero+=int(float(norm)>0)
                optimizer.step();values.append(float(value.detach()))
                if start%80==0:progress(phase='plain_control_training',epoch=epoch+1,epochs=epochs,completed=min(start+8,count),total=count,loss=values[-1],loss_parts=parts)
            history.append(dict(additional_epoch=epoch+1,lr=lr,mean_loss=sum(values)/len(values)));save(out/'history.json',history)
        head.eval();changed=sum(not torch.equal(v,head.state_dict()[k]) for k,v in initial.items());assert changed and nonzero
        checkpoint=out/'last_head.pt';torch.save(dict(state_dict=head.state_dict(),input_size=INPUT,grid=GRID,encoder_sha256=previous['encoder_sha256'],epochs=total,pixel_branch=True),checkpoint)
        scores=None;qualifies=None
        if args.mode=='full':
            metrics={k:0 for k in ('tp','unmatched','fn','predictions','targets')};cases=[]
            with torch.inference_mode():
                for start in range(0,576,8):
                    batch=[torch.load(OLD/'features/inner_val'/f'{i:04d}.pt',map_location='cpu',weights_only=True) for i in range(start,min(start+8,576))]
                    features={k:torch.stack([v['features'][k] for v in batch]) for k in ('semantic','rgb')}
                    predictions=decode(head(features),[v['shape'] for v in batch],score=CROP_SCORE,max_boxes=10)
                    for offset,(item,selected) in enumerate(zip(batch,predictions)):
                        value=metric(selected,item['boxes'])
                        for k,v in value.items():metrics[k]+=v
                        cases.append(dict(crop_index=start+offset,source_image=item['source_image'],metrics=value,predictions=selected))
                    progress(phase='plain_inner_crop_feasibility',completed=min(start+8,576),total=576)
            precision=metrics['tp']/max(1,metrics['tp']+metrics['unmatched']);recall=metrics['tp']/max(1,metrics['targets'])
            scores=dict(metrics=metrics,precision=precision,recall=recall);qualifies=precision>=MIN_CROP_PRECISION and recall>=MIN_CROP_RECALL
            save(out/'inner_crop_evaluation.json',dict(score=CROP_SCORE,summary=scores,cases=cases,repeated_boxes_across_crops=True,source_accuracy=False))
        assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
        result=dict(status='complete',completed_epochs=total,additional_epochs=epochs,head_parameters_changed=changed,nonzero_gradient_steps=nonzero,
            finite_gradients=True,augmentation=False,frozen_encoder_unchanged=True,encoder_sha256=previous['encoder_sha256'],runtime_fingerprint=frozen,
            checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)),crop_feasibility=scores,qualifies_crop_feasibility=qualifies,
            seconds=round(time.monotonic()-started,2),production_changed=False,field_accuracy=False)
        save(out/'report.json',result);save(out/'progress.json',dict(status='complete',qualifies_crop_feasibility=qualifies,seconds=result['seconds']))
        print(str({k:v for k,v in result.items() if k!='runtime_fingerprint'}),flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
