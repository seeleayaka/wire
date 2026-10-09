"""Fixed offline TRAIN-only candidate, finite gradients and input fingerprints."""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('E:/PythonProject10')
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha
BASE=ROOT/'artifacts/allport480_training_20261004'
DATASET=BASE/'dataset'
WEIGHT=REPO/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
TEACHER='9ed5ae77c940a78869e084639c55294b05fc65a78e3723f84af2fdc8189ed824'


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('smoke','full'))
    parser.add_argument('--attempt',default=None,help='fresh smoke directory after a startup failure')
    args=parser.parse_args()
    assert args.attempt is None or (args.mode=='smoke' and args.attempt in ('smoke_v2','smoke_v3'))
    output=BASE/(args.attempt or args.mode)
    if output.exists():
        raise FileExistsError('fresh training output required')
    manifest_path=DATASET/'dataset_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['status']=='complete' and manifest['train_crops']==975
    assert sha(WEIGHT)==TEACHER
    assert sha(BASE/'PLAN.md')==manifest['plan_sha256']
    assert all(sha(Path(p))==v for p,v in manifest['protected_pins'].items())
    for r in manifest['records']:
        for k in ('image','label'):
            assert sha(DATASET/r[k])==r[k+'_sha256']
    if args.mode=='full':
        smoke=json.loads((BASE/'smoke_v3/report.json').read_text(encoding='utf-8'))
        assert smoke['status']=='complete' and smoke['nonzero_gradient_steps']>0
        assert smoke['changed_trainable_parameters']>0 and smoke['dataset_manifest_sha256']==sha(manifest_path)
    chosen=[]
    for positive in (True,False):
        used=set()
        for r in sorted(manifest['records'],key=lambda r:(r['role']!='allport480_positive',r['source_image'],r['image'])):
            if (r['label_count']>0)!=positive or r['source_image'] in used:
                continue
            chosen.append(r);used.add(r['source_image'])
            if len(used)==2:
                break
        assert len(used)==2
    assert len(chosen)==4
    (output/'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',output/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(output/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_DISABLED='True')
    import torch
    import psutil
    from ultralytics import YOLO,settings
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    assert psutil.virtual_memory().available>=5*2**30,'wait for memory without relaxing guard'
    torch.set_num_threads(4)
    settings.update({k:False for k in ('sync','hub','wandb','mlflow','clearml','comet','dvc','neptune','raytune','tensorboard') if k in settings})
    smoke_data=output/'loader_data'
    for r in chosen:
        for k in ('image','label'):
            dest=smoke_data/r[k]
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(DATASET/r[k],dest)
            assert sha(dest)==r[k+'_sha256']
    data_yaml=output/'data.yaml'
    train_path=smoke_data/'images/train' if args.mode=='smoke' else DATASET/'images/train'
    val_path=smoke_data/'images/train'
    data_yaml.write_text('path: '+output.as_posix()+'\ntrain: '+train_path.as_posix()+'\nval: '+val_path.as_posix()+
        '\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n',encoding='utf-8')
    opts=dict(data=str(data_yaml),task='segment',epochs=1 if args.mode=='smoke' else 2,imgsz=960,batch=1,
        nbs=4 if args.mode=='smoke' else 16,workers=0,device='cpu',pretrained=True,freeze=10,
        optimizer='SGD',lr0=.0005,lrf=.1,cos_lr=False,warmup_epochs=.5,warmup_bias_lr=.001,warmup_momentum=.8,
        amp=False,cache=False,deterministic=True,seed=20261004,patience=2,mosaic=0.,close_mosaic=0,
        degrees=0.,translate=.02,scale=.05,fliplr=0.,flipud=0.,plots=False,save=True,val=False,
        project=str(output/'runs'),name='rectports',exist_ok=False,verbose=False,resume=False)
    started=time.monotonic()
    progress=dict(status='starting',mode=args.mode,pid=os.getpid(),options=opts,
        dataset_manifest_sha256=sha(manifest_path),plan_sha256=sha(BASE/'PLAN.md'),
        initialization_sha256=TEACHER,script_sha256=sha(Path(__file__)),
        nonzero_gradient_steps=0,finite_gradients=True,heldout_images_or_labels_used=False,
        diagnostic_val_sources=[r['source_image'] for r in chosen],
        final_checkpoint_policy='last.pt',production_changed=False,wall_deadline_seconds=7200)
    save(output/'progress.json',progress)
    def update(trainer,phase):
        elapsed=time.monotonic()-started
        if elapsed>7200:
            raise TimeoutError('fixed training wall deadline reached')
        progress.update(status='running',phase=phase,epoch=getattr(trainer,'epoch',-1)+1,
            batch=getattr(trainer,'audit_batch',0),batches_per_epoch=len(trainer.train_loader),seconds=round(elapsed,2))
        save(output/'progress.json',progress)
    def hashes(trainer):
        return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                for n,p in trainer.model.named_parameters() if p.requires_grad}
    class AuditedTrainer(SegmentationTrainer):
        def optimizer_step(self):
            norms=[]
            for p in self.model.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(),'nonfinite_gradient'
                    norms.append(float(torch.linalg.vector_norm(p.grad.detach())))
            norm=math.sqrt(sum(v*v for v in norms))
            assert math.isfinite(norm) and norm>0,'invalid_gradient'
            progress['nonzero_gradient_steps']+=1
            super().optimizer_step()
    def on_start(trainer):
        assert len(trainer.train_loader)==(4 if args.mode=='smoke' else 975)
        assert trainer.device.type=='cpu'
        assert len(trainer.test_loader.dataset)==4
        trainer.initial_hashes=hashes(trainer)
        trainer.audit_batch=0
        update(trainer,'training')
    def on_epoch(trainer):
        trainer.audit_batch=0
        update(trainer,'training')
    def on_batch(trainer):
        trainer.audit_batch+=1
        items=trainer.loss_items
        values=list(items.values()) if isinstance(items,dict) else items.flatten()
        assert all(math.isfinite(float(v)) for v in values),'nonfinite_loss'
        if trainer.audit_batch==1 or trainer.audit_batch%10==0 or args.mode=='smoke':
            update(trainer,'training')
    def on_end(trainer):
        final=hashes(trainer)
        progress['changed_trainable_parameters']=sum(final[n]!=v for n,v in trainer.initial_hashes.items())
        assert progress['changed_trainable_parameters']>0
    try:
        model=YOLO(str(WEIGHT))
        assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
        for name,fn in (('on_train_start',on_start),('on_train_epoch_start',on_epoch),
                        ('on_train_batch_end',on_batch),('on_train_end',on_end)):
            model.add_callback(name,fn)
        model.train(trainer=AuditedTrainer,**opts)
        assert progress['nonzero_gradient_steps']>0
        assert sha(WEIGHT)==TEACHER and sha(manifest_path)==progress['dataset_manifest_sha256']
        assert sha(BASE/'PLAN.md')==progress['plan_sha256'] and sha(Path(__file__))==progress['script_sha256']
        assert all(sha(Path(p))==v for p,v in manifest['protected_pins'].items())
        assert model.trainer.epoch+1==opts['epochs']
        weight_path=output/'runs/rectports/weights/last.pt'
        progress.update(status='complete',seconds=round(time.monotonic()-started,2),
            completed_epochs=model.trainer.epoch+1,last_checkpoint=str(weight_path),last_sha256=sha(weight_path),
            warning='Training diagnostics only; original-image source accuracy not evaluated.')
        save(output/'report.json',progress);save(output/'progress.json',progress)
        print(json.dumps({k:progress[k] for k in ('status','mode','completed_epochs','seconds','nonzero_gradient_steps','changed_trainable_parameters')},indent=2),flush=True)
    except BaseException as exc:
        progress.update(status='failed',error=type(exc).__name__+': '+str(exc))
        save(output/'progress.json',progress)
        raise


if __name__=='__main__':
    main()
