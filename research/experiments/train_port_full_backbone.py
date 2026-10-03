"""Fixed two-epoch full-backbone experiment, preserve all deployed checkpoints."""
import argparse,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
BASE=ROOT/'artifacts/port_full_backbone_20261003'
PRIOR=ROOT/'artifacts/port_training_multiscale_20261002';DATASET=PRIOR/'dataset'
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.feature_residual_port_support import WEIGHT_RELATIVE,WEIGHT_SHA,residual_runtime_fingerprint
from train_port_multiscale import smoke_records
from train_port_feature_adaptation import finite_loss_values
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('smoke','full'),required=True);args=parser.parse_args()
    output=BASE/('smoke_v2' if args.mode=='smoke' else 'full')
    if output.exists():raise FileExistsError('Never overwrite a training experiment')
    manifest=load(DATASET/'dataset_manifest.json');audit=load(PRIOR/'audit_v2/report.json')
    assert manifest['status']=='complete' and manifest['source_group_disjoint'] and manifest['train_sources']==192 and manifest['train_crops']==656
    assert audit['status']=='complete' and audit['validation_byte_identical']
    weight=REPO/WEIGHT_RELATIVE;assert sha(weight)==WEIGHT_SHA
    for row in manifest['records']:
        for kind in ('image','label'):assert sha(DATASET/row[kind])==row[kind+'_sha256']
    frozen=residual_runtime_fingerprint(REPO)
    if args.mode=='full':
        smoke=load(BASE/'smoke_v2/report.json')
        assert smoke['status']=='complete' and smoke['changed_early_backbone_tensors']>0 and smoke['frozen_parameters_changed']==0
        assert smoke['script_sha256']==sha(Path(__file__))
    (output/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',output/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(output/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_DISABLED='True')
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
        nbs=4 if args.mode=='smoke' else 16,workers=0,device='cpu',pretrained=True,freeze=0,optimizer='SGD',
        lr0=.0001,lrf=.1,cos_lr=False,warmup_epochs=.5,warmup_bias_lr=.0002,warmup_momentum=.8,
        amp=False,cache=False,deterministic=True,seed=20261003,patience=0,mosaic=0.,close_mosaic=0,
        degrees=0.,translate=.02,scale=.05,fliplr=0.,flipud=0.,plots=False,save=True,val=True,
        project=str(output/'runs'),name='rectports',exist_ok=False,verbose=False,resume=False)
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('train_port_multiscale.py'),
        Path(__file__).with_name('train_port_feature_adaptation.py'),DATASET/'dataset_manifest.json',PRIOR/'audit_v2/report.json',weight)}
    protocol=dict(pins=pins,runtime_fingerprint=frozen,options=opts,
        rationale='Training missed-port geometry/low scores remain the bottleneck; previousfreeze6 did not adapt early0-5 features',
        initialize_from_accepted_feature_last_sha256=WEIGHT_SHA,all_backbone_trainable=True,
        fixed_last_checkpoint=True,no_early_stopping_no_best_checkpoint_selection=True,
        audited_train656_inner576_crops_192_48_sources=True,rectangle_masks_not_real_instance_truth=True,
        source_localization_acceptance_required=True,no_deployment=True,field_accuracy=False)
    save(output/'protocol.json',protocol)
    started=time.monotonic();progress=dict(status='starting',mode=args.mode,pid=os.getpid(),epochs_planned=opts['epochs'],
        options=opts,pins=pins,runtime_fingerprint=frozen,script_sha256=sha(Path(__file__)),
        nonzero_gradient_steps=0,finite_gradients=True,epoch_records=[],sampled_max_rss_GiB=0.,production_weights_changed=False)
    save(output/'progress.json',progress)
    def hashes(trainer):return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in trainer.model.named_parameters()}
    def checkpoint(trainer,phase):
        progress.update(status='running',phase=phase,epoch=trainer.epoch+1,batches_in_epoch=getattr(trainer,'audit_batches',0),
            batches_per_epoch=len(trainer.train_loader),elapsed_seconds=round(time.monotonic()-started,2),cpu_threads=torch.get_num_threads())
        progress['sampled_max_rss_GiB']=max(progress['sampled_max_rss_GiB'],psutil.Process().memory_info().rss/2**30)
        save(output/'progress.json',progress)
    class AuditedTrainer(SegmentationTrainer):
        def optimizer_step(self):
            norms=[]
            for p in self.model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite_gradient')
                    norms.append(float(torch.linalg.vector_norm(p.grad.detach())))
            norm=math.sqrt(sum(v*v for v in norms))
            if not math.isfinite(norm) or norm<=0:raise FloatingPointError('zero_or_invalid_gradient')
            progress['nonzero_gradient_steps']+=1;super().optimizer_step()
    def on_start(trainer):
        torch.set_num_threads(4);trainer.audit_initial=hashes(trainer)
        trainer.audit_frozen={n for n,p in trainer.model.named_parameters() if not p.requires_grad}
        trainer.audit_early={n for n,p in trainer.model.named_parameters() if p.requires_grad and n.split('.')[1] in ('0','1','2','3','4','5')}
        assert trainer.audit_early
        assert len(trainer.train_loader)==(4 if args.mode=='smoke' else 656)
        assert len(trainer.test_loader.dataset)==(2 if args.mode=='smoke' else 576)
    def epoch_start(trainer):torch.set_num_threads(4);trainer.audit_batches=0;checkpoint(trainer,'training')
    def batch_end(trainer):
        trainer.audit_batches+=1;finite_loss_values(trainer.loss_items)
        if trainer.audit_batches==1 or trainer.audit_batches%10==0 or args.mode=='smoke':checkpoint(trainer,'training')
    def epoch_end(trainer):checkpoint(trainer,'inner_crop_validation')
    def fit_end(trainer):
        torch.set_num_threads(4);values={k:float(v) for k,v in trainer.metrics.items()}
        if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite_validation_metric')
        progress['epoch_records'].append(dict(epoch_callback=trainer.epoch+1,metrics=values));checkpoint(trainer,'epoch_finished')
    def train_end(trainer):
        changed={n for n,digest in hashes(trainer).items() if digest!=trainer.audit_initial[n]}
        progress.update(changed_parameter_tensors=len(changed),changed_early_backbone_tensors=len(changed&trainer.audit_early),
            frozen_parameters_changed=len(changed&trainer.audit_frozen))
        assert progress['changed_early_backbone_tensors']>0 and progress['frozen_parameters_changed']==0
    model=YOLO(str(weight));assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    for event,fn in (('on_train_start',on_start),('on_train_epoch_start',epoch_start),('on_train_batch_end',batch_end),
        ('on_train_epoch_end',epoch_end),('on_fit_epoch_end',fit_end),('on_train_end',train_end)):model.add_callback(event,fn)
    try:
        model.train(trainer=AuditedTrainer,**opts)
        assert progress['nonzero_gradient_steps']>0 and {p:sha(Path(p)) for p in pins}==pins and residual_runtime_fingerprint(REPO)==frozen
        assert model.trainer.epoch+1==opts['epochs'];weights=output/'runs/rectports/weights'
        progress.update(status='complete',phase='complete',completed_epochs=opts['epochs'],elapsed_seconds=round(time.monotonic()-started,2),
            weights={n:dict(path=str(weights/n),sha256=sha(weights/n)) for n in ('best.pt','last.pt')},
            warning='Fixedlast only; crop validation is not source/field accuracy; no deployment')
        save(output/'report.json',progress);save(output/'progress.json',progress)
    except BaseException as error:
        progress.update(status='failed',error=type(error).__name__+': '+str(error));save(output/'progress.json',progress);raise
if __name__=='__main__':main()
