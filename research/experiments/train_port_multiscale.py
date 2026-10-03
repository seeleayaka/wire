"""Bounded fine-tune from accepted teacher; candidate weights never replace production."""
import argparse,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
BASE=ROOT/'artifacts/port_training_multiscale_20261002';DATASET=BASE/'dataset'
SMOKE_OUTPUT=BASE/'smoke_v2'
WEIGHT=REPO/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(p)

def smoke_records(records):
    result={}
    for split,pos_count,neg_count in (('train',2,2),('val',1,1)):
        chosen=[];used=set()
        for positive,number in ((True,pos_count),(False,neg_count)):
            candidates=[r for r in records if r['split']==split and (r['label_count']>0)==positive]
            candidates.sort(key=lambda r:(0 if r['role'] in ('multiscale_positive','teacher_hard_negative') else 1,r['source_image'],r['image']))
            for r in candidates:
                if r['source_image'] in used:continue
                chosen.append(r);used.add(r['source_image']);number-=1
                if not number:break
            assert not number
        result[split]=chosen
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True);args=parser.parse_args()
    output=SMOKE_OUTPUT if args.mode=='smoke' else BASE/'full'
    if output.exists():raise FileExistsError('Fresh output required; never overwrite/resume implicitly')
    manifest=load(DATASET/'dataset_manifest.json');protocol=load(BASE/'protocol.json')
    assert manifest['status']=='complete' and manifest['source_group_disjoint'] and manifest['train_sources']==192
    assert sha(WEIGHT)==protocol['teacher_weight_sha256'] and sha(DATASET.parent/'protocol.json')
    for row in manifest['records']:
        for kind in ('image','label'):assert sha(DATASET/row[kind])==row[kind+'_sha256']
    if args.mode=='full':
        audit=load(BASE/'audit_v2/report.json')
        assert audit['status']=='complete' and audit['validation_byte_identical']
        assert audit['counts']['positive_geometry_replayed']==123
        assert audit['counts']['negative_source_annotation_overlap_checked']==64
        acceptance=load(BASE/'evaluation_protocol.json')
        assert acceptance['checkpoint']=='last.pt' and acceptance['freeze_before_training']
        smoke=load(SMOKE_OUTPUT/'report.json')
        assert smoke['status']=='complete' and smoke['nonzero_gradient_steps']>0 and smoke['changed_trainable_parameters']>0
        assert smoke['dataset_manifest_sha256']==sha(DATASET/'dataset_manifest.json')
        assert smoke['initialization_sha256']==sha(WEIGHT)
    (output/'config/Ultralytics').mkdir(parents=True)
    font=Path('C:/Windows/Fonts/arial.ttf');assert font.is_file();shutil.copy2(font,output/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(output/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_DISABLED='True')
    import torch,psutil,ultralytics
    from ultralytics import YOLO,settings
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    torch.set_num_threads(4)
    settings.update({k:False for k in ('sync','hub','wandb','mlflow','clearml','comet','dvc','neptune','raytune','tensorboard') if k in settings})
    data_yaml=DATASET/'data.yaml';selection=None
    if args.mode=='smoke':
        selection=smoke_records(manifest['records']);target=output/'data';classes=set()
        for split,rows in selection.items():
            for row in rows:
                for kind in ('image','label'):
                    dest=target/row[kind];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(DATASET/row[kind],dest)
                if split=='train':classes.update(int(line.split()[0]) for line in (DATASET/row['label']).read_text().splitlines())
        assert classes=={0,1},'Smoke must cover both supervised classes'
        data_yaml=target/'data.yaml';data_yaml.write_text('path: '+target.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n',encoding='utf-8')
    opts=dict(data=str(data_yaml),task='segment',epochs=1 if args.mode=='smoke' else 2,imgsz=960,batch=1,
        nbs=4 if args.mode=='smoke' else 16,workers=0,device='cpu',pretrained=True,freeze=10,
        optimizer='SGD',lr0=.0005,lrf=.1,cos_lr=False,warmup_epochs=.5,warmup_bias_lr=.001,warmup_momentum=.8,
        amp=False,cache=False,deterministic=True,seed=20261002,patience=2,mosaic=0.,close_mosaic=0,
        degrees=0.,translate=.02,scale=.05,fliplr=0.,flipud=0.,plots=False,save=True,val=True,
        project=str(output/'runs'),name='rectports',exist_ok=False,verbose=False,resume=False)
    started=time.monotonic();progress=dict(status='starting',mode=args.mode,pid=os.getpid(),epochs_planned=opts['epochs'],
        dataset_manifest_sha256=sha(DATASET/'dataset_manifest.json'),initialization_sha256=sha(WEIGHT),protocol_sha256=sha(BASE/'protocol.json'),
        script_sha256=sha(Path(__file__)),options=opts,smoke_selection=selection,finite_gradients=True,nonzero_gradient_steps=0,
        sampled_max_rss_GiB=0.,epoch_records=[],versions=dict(torch=torch.__version__,ultralytics=ultralytics.__version__),
        select_final_last_checkpoint=True,production_weights_changed=False,outer_val_test_used=False)
    if args.mode=='full':
        progress['evaluation_protocol_sha256']=sha(BASE/'evaluation_protocol.json')
        progress['dataset_audit_sha256']=sha(BASE/'audit_v2/report.json')
    save(output/'progress.json',progress)
    def event(kind,values):
        with (output/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(event=kind,seconds=time.monotonic()-started,**values))+'\n')
    def checkpoint(trainer,phase):
        progress.update(status='running',phase=phase,epoch=trainer.epoch+1,batches_in_epoch=getattr(trainer,'audit_batches',0),
            batches_per_epoch=len(trainer.train_loader),elapsed_seconds=round(time.monotonic()-started,2),cpu_threads=torch.get_num_threads())
        progress['sampled_max_rss_GiB']=max(progress['sampled_max_rss_GiB'],psutil.Process().memory_info().rss/2**30)
        save(output/'progress.json',progress)
    def parameter_hashes(trainer):return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in trainer.model.named_parameters() if p.requires_grad}
    class AuditedTrainer(SegmentationTrainer):
        def optimizer_step(self):
            norms=[]
            for p in self.model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite_gradient')
                    norms.append(float(torch.linalg.vector_norm(p.grad.detach())))
            norm=math.sqrt(sum(v*v for v in norms))
            if not math.isfinite(norm) or norm<=0:raise FloatingPointError('zero_or_invalid_gradient')
            progress['nonzero_gradient_steps']+=1;event('optimizer_step',dict(epoch=self.epoch+1,raw_gradient_l2=norm))
            super().optimizer_step()
    def on_start(trainer):
        torch.set_num_threads(4)
        assert len(trainer.train_loader)==(4 if args.mode=='smoke' else manifest['train_crops'])
        assert len(trainer.test_loader.dataset)==(2 if args.mode=='smoke' else 576)
        assert trainer.device.type=='cpu';trainer.audit_initial_hashes=parameter_hashes(trainer)
        event('train_start',dict(train_crops=len(trainer.train_loader),val_crops=len(trainer.test_loader.dataset)))
    def epoch_start(trainer):
        torch.set_num_threads(4)
        trainer.audit_batches=0;checkpoint(trainer,'training')
    def batch_end(trainer):
        trainer.audit_batches+=1;items=trainer.loss_items
        if isinstance(items,dict):values={k:float(v.detach().cpu()) for k,v in items.items()}
        else:values={str(i):float(v.detach().cpu()) for i,v in enumerate(items.flatten())}
        if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite_loss')
        if trainer.audit_batches==1 or trainer.audit_batches%10==0 or args.mode=='smoke':
            event('batch',dict(epoch=trainer.epoch+1,batch=trainer.audit_batches,losses=values));checkpoint(trainer,'training')
    def epoch_end(trainer):checkpoint(trainer,'inner_crop_validation')
    def fit_end(trainer):
        torch.set_num_threads(4)
        values={k:float(v) for k,v in trainer.metrics.items()}
        if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite_inner_metric')
        progress['epoch_records'].append(dict(epoch=trainer.epoch+1,metrics=values));checkpoint(trainer,'epoch_finished')
    def train_end(trainer):
        final=parameter_hashes(trainer);progress['changed_trainable_parameters']=sum(final[k]!=v for k,v in trainer.audit_initial_hashes.items())
        assert progress['changed_trainable_parameters']>0
    model=YOLO(str(WEIGHT))
    assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    for e,fn in (('on_train_start',on_start),('on_train_epoch_start',epoch_start),('on_train_batch_end',batch_end),
        ('on_train_epoch_end',epoch_end),('on_fit_epoch_end',fit_end),('on_train_end',train_end)):model.add_callback(e,fn)
    try:
        model.train(trainer=AuditedTrainer,**opts)
        assert progress['nonzero_gradient_steps']>0 and sha(WEIGHT)==progress['initialization_sha256']
        assert sha(DATASET/'dataset_manifest.json')==progress['dataset_manifest_sha256']
        assert sha(Path(__file__))==progress['script_sha256'] and sha(BASE/'protocol.json')==progress['protocol_sha256']
        if args.mode=='full':
            assert sha(BASE/'evaluation_protocol.json')==progress['evaluation_protocol_sha256']
            assert sha(BASE/'audit_v2/report.json')==progress['dataset_audit_sha256']
        weights=output/'runs/rectports/weights'
        progress.update(status='complete',phase='complete',epoch=model.trainer.epoch+1,completed_epochs=model.trainer.epoch+1,elapsed_seconds=round(time.monotonic()-started,2),
            weights={n:dict(path=str(weights/n),sha256=sha(weights/n)) for n in ('best.pt','last.pt')},
            warning='Internal crop metrics are not original-image localization, physical fault or general-cabinet accuracy.')
        assert progress['completed_epochs']==opts['epochs']
        save(output/'report.json',progress);save(output/'progress.json',progress)
        print(json.dumps({k:v for k,v in progress.items() if k in ('status','mode','completed_epochs','nonzero_gradient_steps','changed_trainable_parameters','elapsed_seconds','sampled_max_rss_GiB')},indent=2),flush=True)
    except BaseException as exc:
        progress.update(status='failed',error=type(exc).__name__+': '+str(exc));save(output/'progress.json',progress);raise

if __name__=='__main__':main()
