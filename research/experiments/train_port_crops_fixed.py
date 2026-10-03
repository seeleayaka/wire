"""Run the frozen smoke/full crop recipe, with finite gradients and durable progress."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--dataset',type=Path,default=Path('E:/PythonProject10/data/derived/port_crop_training_20260929'))
    p.add_argument('--weights',type=Path,default=Path('E:/wire_harness_training_bundle/weights/yolov8s-seg.pt'))
    p.add_argument('--mode',choices=('smoke','full'),required=True)
    p.add_argument('--smoke-report',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required, no implicit resume')
    args.output=args.output.resolve();sys.path.insert(0,str(args.repo))
    from inspection_agent.port_crop_training import BASE_SHA256,training_options,select_smoke_records
    assert sha(args.weights)==BASE_SHA256,'initialization must be the verified generic local base'
    manifest_path=args.dataset/'dataset_manifest.json';dataset=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert dataset['status']=='complete' and dataset['source_split']=='train01' and dataset['source_group_disjoint']
    assert len(dataset['records'])==1045 and dataset['train_crops']==469 and dataset['inner_val_crops']==576
    if args.mode=='full':
        if not args.smoke_report:raise ValueError('successful smoke report required')
        smoke=json.loads(args.smoke_report.read_text(encoding='utf-8'))
        assert smoke['status']=='complete' and smoke['mode']=='smoke' and smoke['finite_gradients'] and smoke['nonzero_gradient_steps']>0
        assert smoke['changed_trainable_parameters']>0 and smoke['dataset_manifest_sha256']==sha(manifest_path)
        assert smoke['initialization_sha256']==BASE_SHA256
    for record in dataset['records']:
        assert sha(args.dataset/record['image'])==record['image_sha256']
        assert sha(args.dataset/record['label'])==record['label_sha256']
    args.output.mkdir(parents=True);(args.output/'config/Ultralytics').mkdir(parents=True)
    font=Path('C:/Windows/Fonts/arial.ttf')
    if not font.is_file():raise FileNotFoundError('local Arial required; do not download fonts')
    shutil.copy2(font,args.output/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(args.output/'config'),YOLO_OFFLINE='true',
                      OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_DISABLED='true')
    import torch
    import psutil
    import ultralytics
    from ultralytics import YOLO,settings
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    torch.set_num_threads(4)
    settings.update({key:False for key in ('sync','hub','wandb','mlflow','clearml','comet','dvc','neptune','raytune','tensorboard') if key in settings})
    data_yaml=args.dataset/'port_crops_rectseg.yaml'
    selected=None
    if args.mode=='smoke':
        selected=select_smoke_records(dataset['records']);smoke_data=args.output/'smoke_dataset'
        classes=set()
        for split,records in selected.items():
            for record in records:
                for kind in ('image','label'):
                    destination=smoke_data/record[kind];destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(args.dataset/record[kind],destination)
                classes.update(int(line.split()[0]) for line in (smoke_data/record['label']).read_text(encoding='utf-8').splitlines())
        assert classes=={0,1},'smoke must exercise both supervised classes'
        data_yaml=smoke_data/'data.yaml'
        data_yaml.write_text('path: '+smoke_data.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n',encoding='utf-8')
    else:
        expected='path: '+args.dataset.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n'
        assert data_yaml.read_text(encoding='utf-8')==expected,'unexpected dataset routing'
    options=training_options(data_yaml,args.output,args.mode);start=time.perf_counter()
    progress={'status':'starting','mode':args.mode,'pid':os.getpid(),'epochs_planned':options['epochs'],
              'dataset_manifest_sha256':sha(manifest_path),'initialization_sha256':BASE_SHA256,
              'options':options,'smoke_selection':selected,
              'fingerprints':{str(path):sha(path) for path in (Path(__file__),args.repo/'inspection_agent/port_crop_training.py',
                              args.repo/'inspection_agent/PORT_CROP_TRAIN_PROTOCOL_20260929.md')},
              'versions':{'torch':torch.__version__,'ultralytics':ultralytics.__version__,'python':sys.version},
              'outer_validation_or_test_used':False,'finite_gradients':True,'nonzero_gradient_steps':0,
              'epoch_records':[],'sampled_max_rss_GiB':0.}
    save(args.output/'progress.json',progress)
    def record(kind,values):
        with (args.output/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'event':kind,'elapsed_seconds':time.perf_counter()-start,**values})+'\n')
    def checkpoint(trainer,phase):
        progress.update(status='running',phase=phase,epoch=trainer.epoch+1,
                        batches_in_epoch=getattr(trainer,'audit_batches',0),
                        batches_per_epoch=len(trainer.train_loader),elapsed_seconds=time.perf_counter()-start)
        progress['sampled_max_rss_GiB']=max(progress['sampled_max_rss_GiB'],psutil.Process().memory_info().rss/2**30)
        save(args.output/'progress.json',progress)
    def parameter_hashes(trainer):
        return {name:hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()
                for name,param in trainer.model.named_parameters() if param.requires_grad}
    class AuditedTrainer(SegmentationTrainer):
        def optimizer_step(self):
            norms=[]
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():raise FloatingPointError('nonfinite gradient')
                    norms.append(float(torch.linalg.vector_norm(parameter.grad.detach())))
            norm=math.sqrt(sum(value*value for value in norms))
            if not math.isfinite(norm) or norm<=0:raise FloatingPointError('invalid or zero accumulated gradient')
            progress['nonzero_gradient_steps']+=1
            record('optimizer_step',{'epoch':self.epoch+1,'raw_gradient_l2':norm})
            super().optimizer_step()
    def on_start(trainer):
        expected=4 if args.mode=='smoke' else 469
        assert len(trainer.train_loader)==expected
        assert len(trainer.test_loader.dataset)==(2 if args.mode=='smoke' else 576)
        assert trainer.device.type=='cpu' and trainer.args.warmup_epochs==1.
        if args.mode=='smoke':trainer.audit_initial_hashes=parameter_hashes(trainer)
        record('train_start',{'train_samples':expected,'val_samples':len(trainer.test_loader.dataset)})
    def epoch_start(trainer):trainer.audit_batches=0;checkpoint(trainer,'training')
    def batch_end(trainer):
        trainer.audit_batches+=1
        values={name:float(value.detach().cpu()) for name,value in trainer.loss_items.items()}
        if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite loss')
        if trainer.audit_batches==1 or trainer.audit_batches%25==0 or args.mode=='smoke':
            record('batch',{'epoch':trainer.epoch+1,'batch':trainer.audit_batches,'losses':values})
            checkpoint(trainer,'training')
    def epoch_end(trainer):checkpoint(trainer,'inner_validation')
    def fit_end(trainer):
        # final_eval may call this again for the same epoch; record it separately.
        values={k:float(v) for k,v in trainer.metrics.items()}
        if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite inner metric')
        progress['epoch_records'].append({'epoch':trainer.epoch+1,'inner_crop_metrics':values,
                                          'elapsed_seconds':time.perf_counter()-start})
        checkpoint(trainer,'epoch_finished')
    def train_end(trainer):
        if args.mode=='smoke':
            final=parameter_hashes(trainer)
            progress['changed_trainable_parameters']=sum(final[k]!=v for k,v in trainer.audit_initial_hashes.items())
            assert progress['changed_trainable_parameters']>0,'smoke performed no parameter update'
    model=YOLO(str(args.weights))
    for event,fn in (('on_train_start',on_start),('on_train_epoch_start',epoch_start),
                     ('on_train_batch_end',batch_end),('on_train_epoch_end',epoch_end),
                     ('on_fit_epoch_end',fit_end),('on_train_end',train_end)):
        model.add_callback(event,fn)
    try:
        model.train(trainer=AuditedTrainer,**options)
        assert progress['nonzero_gradient_steps']>0
        for path,digest in progress['fingerprints'].items():assert sha(Path(path))==digest,'running code/protocol drift'
        assert sha(args.weights)==BASE_SHA256 and sha(manifest_path)==progress['dataset_manifest_sha256']
        run=args.output/'runs/rectports'
        progress.update(status='complete',phase='complete',elapsed_seconds=time.perf_counter()-start,
                        completed_epochs=model.trainer.epoch+1,
                        weights={name:{'path':str(run/'weights'/name),'sha256':sha(run/'weights'/name)} for name in ('best.pt','last.pt')},
                        warning='Internal tile metrics only; not source-image localization, external val accuracy or field/generalization evidence.')
        assert progress['completed_epochs']==options['epochs']
        save(args.output/'report.json',progress);save(args.output/'progress.json',progress)
        print(json.dumps({k:v for k,v in progress.items() if k in ('status','mode','completed_epochs','nonzero_gradient_steps','changed_trainable_parameters','elapsed_seconds','sampled_max_rss_GiB')},indent=2))
    except BaseException as exc:
        progress.update(status='failed',error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.perf_counter()-start)
        save(args.output/'progress.json',progress)
        raise


if __name__=='__main__':main()
