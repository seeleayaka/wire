"""Bounded feature adaptation from the audited student, no mainline replacement."""
import argparse,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');BASE=ROOT/'artifacts/port_feature_adaptation_20261003'
PRIOR=ROOT/'artifacts/port_training_multiscale_20261002';DATASET=PRIOR/'dataset'
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.teacher_student_port_support import STUDENT_SHA,STUDENT_RELATIVE,TEACHER_SHA,TEACHER_RELATIVE
from train_port_multiscale import smoke_records
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def finite_loss_values(items):
    values=({str(k):float(v.detach().cpu()) for k,v in items.items()} if isinstance(items,dict)
        else {str(i):float(v.detach().cpu()) for i,v in enumerate(items.flatten())})
    if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite_loss')
    return values
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True);args=parser.parse_args()
    output=BASE/('smoke_v2' if args.mode=='smoke' else 'full')
    if output.exists():raise FileExistsError('Fresh output required')
    protocol=load(BASE/'protocol.json');manifest=load(DATASET/'dataset_manifest.json');audit=load(PRIOR/'audit_v2/report.json')
    assert manifest['status']=='complete' and manifest['source_group_disjoint'] and manifest['train_sources']==192 and manifest['train_crops']==656
    assert audit['status']=='complete' and audit['validation_byte_identical']
    assert sha(REPO/STUDENT_RELATIVE)==STUDENT_SHA and sha(REPO/TEACHER_RELATIVE)==TEACHER_SHA
    assert sha(DATASET/'dataset_manifest.json')==protocol['dataset_manifest_sha256']
    assert protocol['fixed_last_checkpoint'] and protocol['epochs']==2 and protocol['freeze']==6
    for row in manifest['records']:
        for kind in ('image','label'):assert sha(DATASET/row[kind])==row[kind+'_sha256']
    if args.mode=='full':
        smoke=load(BASE/'smoke_v2/report.json');assert smoke['status']=='complete'
        assert smoke['nonzero_gradient_steps']>0 and smoke['changed_new_backbone_tensors']>0 and smoke['frozen_parameters_changed']==0
        assert smoke['script_sha256']==sha(Path(__file__)) and smoke['protocol_sha256']==sha(BASE/'protocol.json')
    (output/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',output/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(output/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_DISABLED='True')
    import torch,psutil
    from ultralytics import YOLO,settings
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    torch.set_num_threads(4)
    settings.update({k:False for k in ('sync','hub','wandb','mlflow','clearml','comet','dvc','neptune','raytune','tensorboard') if k in settings})
    yaml=DATASET/'data.yaml';selection=None
    if args.mode=='smoke':
        selection=smoke_records(manifest['records']);target=output/'data'
        for split,rows in selection.items():
            for row in rows:
                for kind in ('image','label'):
                    dest=target/row[kind];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(DATASET/row[kind],dest)
        yaml=target/'data.yaml';yaml.write_text('path: '+target.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n',encoding='utf-8')
    opts=dict(data=str(yaml),task='segment',epochs=1 if args.mode=='smoke' else 2,imgsz=960,batch=1,
        nbs=4 if args.mode=='smoke' else 16,workers=0,device='cpu',pretrained=True,freeze=6,
        optimizer='SGD',lr0=.00025,lrf=.1,cos_lr=False,warmup_epochs=.5,warmup_bias_lr=.0005,warmup_momentum=.8,
        amp=False,cache=False,deterministic=True,seed=20261003,patience=0,mosaic=0.,close_mosaic=0,
        degrees=0.,translate=.02,scale=.05,fliplr=0.,flipud=0.,plots=False,save=True,val=True,
        project=str(output/'runs'),name='rectports',exist_ok=False,verbose=False,resume=False)
    started=time.monotonic();pins={str(p):sha(p) for p in (Path(__file__),BASE/'protocol.json',DATASET/'dataset_manifest.json',PRIOR/'audit_v2/report.json',REPO/STUDENT_RELATIVE,REPO/TEACHER_RELATIVE,REPO/'config/port_crop_calibration_frozen_20260930.json')}
    progress=dict(status='starting',mode=args.mode,pid=os.getpid(),epochs_planned=opts['epochs'],options=opts,pins=pins,
        initialization_sha256=STUDENT_SHA,script_sha256=sha(Path(__file__)),protocol_sha256=sha(BASE/'protocol.json'),
        dataset_manifest_sha256=sha(DATASET/'dataset_manifest.json'),smoke_selection=selection,nonzero_gradient_steps=0,
        finite_gradients=True,epoch_records=[],sampled_max_rss_GiB=0.,production_weights_changed=False,
        outer_val_test_used=False,frozen_parameters_not_batchnorm_buffers=True)
    save(output/'progress.json',progress)
    def event(kind,values):
        with (output/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(event=kind,seconds=time.monotonic()-started,**values))+'\n')
    def checkpoint(trainer,phase):
        progress.update(status='running',phase=phase,epoch=trainer.epoch+1,batches_in_epoch=getattr(trainer,'audit_batches',0),
            batches_per_epoch=len(trainer.train_loader),elapsed_seconds=round(time.monotonic()-started,2),cpu_threads=torch.get_num_threads())
        progress['sampled_max_rss_GiB']=max(progress['sampled_max_rss_GiB'],psutil.Process().memory_info().rss/2**30);save(output/'progress.json',progress)
    def hashes(trainer):return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in trainer.model.named_parameters()}
    class AuditedTrainer(SegmentationTrainer):
        def optimizer_step(self):
            norms=[]
            for p in self.model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite_gradient')
                    norms.append(float(torch.linalg.vector_norm(p.grad.detach())))
            norm=math.sqrt(sum(v*v for v in norms))
            if not math.isfinite(norm) or norm<=0:raise FloatingPointError('zero_or_invalid_gradient')
            progress['nonzero_gradient_steps']+=1;event('optimizer_step',dict(epoch=self.epoch+1,raw_gradient_l2=norm));super().optimizer_step()
    def on_start(trainer):
        torch.set_num_threads(4);trainer.audit_initial_hashes=hashes(trainer)
        trainer.audit_frozen={n for n,p in trainer.model.named_parameters() if not p.requires_grad}
        trainer.audit_new_backbone={n for n,p in trainer.model.named_parameters() if p.requires_grad and n.split('.')[1] in ('6','7','8','9')}
        assert trainer.audit_new_backbone
        assert len(trainer.train_loader)==(4 if args.mode=='smoke' else 656)
        assert len(trainer.test_loader.dataset)==(2 if args.mode=='smoke' else 576)
        event('train_start',dict(train_crops=len(trainer.train_loader),newly_trainable_backbone_tensors=len(trainer.audit_new_backbone)))
    def epoch_start(trainer):torch.set_num_threads(4);trainer.audit_batches=0;checkpoint(trainer,'training')
    def batch_end(trainer):
        trainer.audit_batches+=1;items=trainer.loss_items
        values=finite_loss_values(items)
        if trainer.audit_batches==1 or trainer.audit_batches%10==0 or args.mode=='smoke':event('batch',dict(epoch=trainer.epoch+1,batch=trainer.audit_batches,losses=values));checkpoint(trainer,'training')
    def epoch_end(trainer):checkpoint(trainer,'inner_crop_validation')
    def fit_end(trainer):
        torch.set_num_threads(4);values={k:float(v) for k,v in trainer.metrics.items()}
        if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite_inner_metric')
        progress['epoch_records'].append(dict(epoch=trainer.epoch+1,metrics=values));checkpoint(trainer,'epoch_finished')
    def train_end(trainer):
        final=hashes(trainer);changed={n for n,v in final.items() if v!=trainer.audit_initial_hashes[n]}
        progress['changed_parameter_tensors']=len(changed);progress['changed_new_backbone_tensors']=len(changed&trainer.audit_new_backbone)
        progress['frozen_parameters_changed']=len(changed&trainer.audit_frozen)
        assert progress['changed_new_backbone_tensors']>0 and progress['frozen_parameters_changed']==0
    model=YOLO(str(REPO/STUDENT_RELATIVE));assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    for e,fn in (('on_train_start',on_start),('on_train_epoch_start',epoch_start),('on_train_batch_end',batch_end),('on_train_epoch_end',epoch_end),('on_fit_epoch_end',fit_end),('on_train_end',train_end)):model.add_callback(e,fn)
    try:
        model.train(trainer=AuditedTrainer,**opts)
        assert progress['nonzero_gradient_steps']>0 and {p:sha(Path(p)) for p in pins}==pins
        weights=output/'runs/rectports/weights';completed=model.trainer.epoch+1;assert completed==opts['epochs']
        progress.update(status='complete',phase='complete',completed_epochs=completed,elapsed_seconds=round(time.monotonic()-started,2),
            weights={n:dict(path=str(weights/n),sha256=sha(weights/n)) for n in ('best.pt','last.pt')},
            warning='Fixed last.pt only; crop metrics are not original-image, field fault or cross-cabinet accuracy')
        save(output/'report.json',progress);save(output/'progress.json',progress)
        print(json.dumps({k:v for k,v in progress.items() if k in ('status','mode','completed_epochs','nonzero_gradient_steps','changed_new_backbone_tensors','frozen_parameters_changed','elapsed_seconds')}),flush=True)
    except BaseException as error:
        progress.update(status='failed',error=type(error).__name__+': '+str(error));save(output/'progress.json',progress);raise
if __name__=='__main__':main()
