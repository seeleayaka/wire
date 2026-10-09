"""Conditional fixed expanded-data backbone continuation. No automatic deployment."""
import argparse
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint

BASE=ROOT/'artifacts/allport480_backbone_continuation_20261005'
PLAN=ROOT/'artifacts/allport480_backbone_continuation_preregistration_20261005/PLAN.md'
TRAIN=ROOT/'artifacts/allport480_training_20261004'
DATASET=TRAIN/'dataset'
SOURCE=ROOT/'artifacts/allport480_teacher_source_20261005'
SOURCE_AUDIT=ROOT/'artifacts/allport480_teacher_source_audit_20261005/report.json'
DEV=ROOT/'artifacts/allport480_teacher_holdouts_20261005'
DEV_AUDIT=ROOT/'artifacts/allport480_teacher_holdout_audit_20261005/report.json'
INIT_SHA='7a893f494eb4e2433712047abcd58fb1e036043da96e31bca6bd0e77bed143ec'
GROUPS=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json'


def entry_allowed(development,audit):
    if (development.get('status')!='rejected' or development.get('failed_stage') not in ('inner','outer')
        or audit.get('status')!='pass' or audit.get('candidate_development_qualifies') is not False):
        raise ValueError('only after completed independently rejected development; protect running/pass branches')
    return True


def fixed_options(mode,data,output):
    if mode not in ('smoke','full'):raise ValueError('unknown training mode')
    return dict(data=str(data),task='segment',epochs=1 if mode=='smoke' else 2,imgsz=960,batch=1,nbs=4 if mode=='smoke' else 16,
        workers=0,device='cpu',pretrained=True,freeze=0,optimizer='SGD',lr0=.0001,lrf=.1,cos_lr=False,
        warmup_epochs=.5,warmup_bias_lr=.0002,warmup_momentum=.8,amp=False,cache=False,deterministic=True,
        seed=20261005,patience=0,mosaic=0.,close_mosaic=0,degrees=0.,translate=.02,scale=.05,fliplr=0.,flipud=0.,
        plots=False,save=True,val=False,project=str(output/'runs'),name='rectports',exist_ok=False,verbose=False,resume=False)


def choose_train_diagnostics(records):
    chosen=[]
    for positive in (True,False):
        used=set()
        for row in sorted(records,key=lambda r:(r['role']!='allport480_positive',r['source_image'],r['image'])):
            if (row['label_count']>0)!=positive or row['source_image'] in used:continue
            chosen.append(row);used.add(row['source_image'])
            if len(used)==2:break
        if len(used)!=2:raise ValueError('TRAIN diagnostics need two distinct sources per label-presence group')
    if len({r['source_image'] for r in chosen})!=4:raise ValueError('diagnostic sources overlap')
    return chosen


def main(mode):
    output=BASE/mode
    if output.exists():raise FileExistsError('never overwrite/resume a training attempt')
    # No detector/training imports, data/GT reads or output creation before this.
    development=load(DEV/'report.json');audit=load(DEV_AUDIT);entry_allowed(development,audit)
    if audit['source_report_sha256']!=sha(DEV/'report.json') or any(sha(p)!=d for p,d in development['pins'].items()):raise ValueError('development evidence changed')
    if any(sha(p)!=d for p,d in audit['source_case_sha256'].items()):raise ValueError('independently audited development outputs changed')
    source=load(SOURCE/'report.json');sourceaudit=load(SOURCE_AUDIT)
    if not source['qualifies'] or sourceaudit['status']!='pass' or not sourceaudit['candidate_source_qualifies'] or sourceaudit['source_report_sha256']!=sha(SOURCE/'report.json'):
        raise ValueError('accepted source300 evidence required')
    frozen=native_pose_runtime_fingerprint(REPO)
    if frozen!=source['runtime'] or frozen!=development['runtime']:raise ValueError('preserve E mainline runtime')
    dirty=lambda:subprocess.check_output(['E:/Git/cmd/git.exe','-C',str(REPO),'status','--porcelain'],text=True)
    before=dirty()
    manifestpath=DATASET/'dataset_manifest.json';manifest=load(manifestpath);groups=load(GROUPS)
    if manifest['status']!='complete' or manifest['train_crops']!=975 or len(manifest['records'])!=975:raise ValueError('fixed audited975 source crops required')
    if len(set(groups['train_sources']))!=192 or set(groups['train_sources'])&set(groups['inner_val_sources']):raise ValueError('TRAIN/INNER membership overlap')
    names=set(groups['train_sources']);trainreport=load(TRAIN/'full/report.json');dataaudit=load(TRAIN/'dataset_audit/report.json')
    if (trainreport['status']!='complete' or trainreport['last_sha256']!=INIT_SHA or trainreport['dataset_manifest_sha256']!=sha(manifestpath)
        or dataaudit['status']!='complete' or dataaudit['manifest_sha256']!=sha(manifestpath)
        or dataaudit['auditor_sha256']!=sha(ROOT/'experiments/audit_allport480_dataset.py')):raise ValueError('previous training/data provenance drift')
    weight=Path(trainreport['last_checkpoint'])
    if sha(weight)!=INIT_SHA or any(sha(p)!=d for p,d in manifest['protected_pins'].items()):raise ValueError('original init/protected sources changed')
    for record in manifest['records']:
        if record['split']!='train' or record['source_image'] not in names:raise ValueError('nonTRAIN crop in training input')
        for kind,folder in [('image','images'),('label','labels')]:
            path=(DATASET/record[kind]).resolve()
            if not path.is_relative_to((DATASET/folder/'train').resolve()) or sha(path)!=record[kind+'_sha256']:raise ValueError('source crop path/bytes changed')
    code=[Path(__file__),PLAN,manifestpath,GROUPS,weight,TRAIN/'full/report.json',TRAIN/'dataset_audit/report.json',
        SOURCE/'report.json',SOURCE_AUDIT,DEV/'report.json',DEV_AUDIT]
    pins={str(p):sha(p) for p in code}
    if mode=='full':
        smokepath=BASE/'smoke/report.json';smoke=load(smokepath);pins[str(smokepath)]=sha(smokepath)
        if (smoke['status']!='complete' or smoke['mode']!='smoke' or smoke['initialization_sha256']!=INIT_SHA
            or smoke['dataset_manifest_sha256']!=sha(manifestpath) or smoke['plan_sha256']!=sha(PLAN)
            or smoke['script_sha256']!=sha(Path(__file__)) or smoke['nonzero_gradient_steps']<=0
            or smoke['changed_early_backbone_tensors']<=0 or smoke['changed_frozen_tensors']!=0):raise ValueError('successful pinned backbone smoke required')
    chosen=choose_train_diagnostics(manifest['records'])
    import torch,psutil
    if psutil.virtual_memory().available<6*2**30:raise RuntimeError('insufficient free memory; do not close clients/compete')
    (output/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',output/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(output/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_DISABLED='True')
    torch.set_num_threads(2)
    from ultralytics import YOLO,settings
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    settings.update({k:False for k in ('sync','hub','wandb','mlflow','clearml','comet','dvc','neptune','raytune','tensorboard') if k in settings})
    diagnostic=output/'train_diagnostics'
    for record in chosen:
        for kind in ('image','label'):
            dest=diagnostic/record[kind];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(DATASET/record[kind],dest)
            if sha(dest)!=record[kind+'_sha256']:raise ValueError('copied TRAIN diagnostic changed')
    datayaml=output/'data.yaml';trainpath=diagnostic/'images/train' if mode=='smoke' else DATASET/'images/train'
    datayaml.write_text('path: '+output.as_posix()+'\ntrain: '+trainpath.as_posix()+'\nval: '+(diagnostic/'images/train').as_posix()+
        '\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n',encoding='utf-8')
    options=fixed_options(mode,datayaml,output);started=time.monotonic()
    progress=dict(status='starting',pid=os.getpid(),mode=mode,options=options,initialization_sha256=INIT_SHA,
        dataset_manifest_sha256=sha(manifestpath),plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
        nonzero_gradient_steps=0,finite_gradients=True,changed_early_backbone_tensors=0,changed_frozen_tensors=0,
        final_checkpoint_policy='last.pt',diagnostic_val_sources=[r['source_image'] for r in chosen],
        heldout_images_or_labels_fitted=False,field_accuracy=None,no_deployment=True,wall_deadline_seconds=14400)
    save(output/'protocol.json',dict(**progress,pins=pins,runtime=frozen,mainline_dirty_before=before))
    save(output/'progress.json',progress)
    def hashes(model):return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
    def update(trainer,phase):
        elapsed=time.monotonic()-started
        if elapsed>14400:raise TimeoutError('operational bound; partial training is not an evaluated model')
        progress.update(status='running',phase=phase,epoch=trainer.epoch+1,batch=getattr(trainer,'audit_batch',0),seconds=round(elapsed,2))
        save(output/'progress.json',progress)
    class AuditedTrainer(SegmentationTrainer):
        def optimizer_step(self):
            norms=[]
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():raise ValueError('nonfinite training gradient')
                    norms.append(float(torch.linalg.vector_norm(parameter.grad.detach())))
            norm=math.sqrt(sum(v*v for v in norms))
            if not math.isfinite(norm) or norm<=0:raise ValueError('finite nonzero gradient required')
            progress['nonzero_gradient_steps']+=1;super().optimizer_step()
    def on_start(trainer):
        if len(trainer.train_loader)!=(4 if mode=='smoke' else 975) or len(trainer.test_loader.dataset)!=4 or trainer.device.type!='cpu':raise ValueError('fixed TRAIN loader contract changed')
        trainer.audit_initial=hashes(trainer.model);trainer.audit_frozen={n for n,p in trainer.model.named_parameters() if not p.requires_grad}
        trainer.audit_early={n for n,p in trainer.model.named_parameters() if n.startswith('model.') and n.split('.')[1].isdigit() and int(n.split('.')[1])<10 and p.requires_grad}
        if not trainer.audit_early:raise ValueError('early backbone not actually trainable')
        trainer.audit_batch=0;update(trainer,'training')
    def on_epoch(trainer):trainer.audit_batch=0;update(trainer,'training')
    def on_batch(trainer):
        trainer.audit_batch+=1;items=trainer.loss_items
        values=list(items.values()) if isinstance(items,dict) else items.flatten()
        if not all(math.isfinite(float(v)) for v in values):raise ValueError('nonfinite training loss')
        if trainer.audit_batch==1 or trainer.audit_batch%10==0 or mode=='smoke':update(trainer,'training')
    def on_end(trainer):
        final=hashes(trainer.model);changed={n for n,v in trainer.audit_initial.items() if final[n]!=v}
        progress.update(changed_trainable_parameters=len(changed-trainer.audit_frozen),changed_early_backbone_tensors=len(changed&trainer.audit_early),changed_frozen_tensors=len(changed&trainer.audit_frozen))
        if progress['changed_early_backbone_tensors']<=0 or progress['changed_frozen_tensors']!=0:raise ValueError('actual backbone/frozen parameter contract failed')
    try:
        model=YOLO(str(weight))
        if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('fixed detector class contract changed')
        for event,fn in [('on_train_start',on_start),('on_train_epoch_start',on_epoch),('on_train_batch_end',on_batch),('on_train_end',on_end)]:model.add_callback(event,fn)
        model.train(trainer=AuditedTrainer,**options)
        if (model.trainer.epoch+1!=options['epochs'] or progress['nonzero_gradient_steps']<=0 or progress['changed_early_backbone_tensors']<=0
            or progress['changed_frozen_tensors']!=0 or any(sha(p)!=d for p,d in pins.items())
            or native_pose_runtime_fingerprint(REPO)!=frozen or dirty()!=before):raise ValueError('complete training/E provenance verification failed')
        for r in manifest['records']:
            if any(sha(DATASET/r[k])!=r[k+'_sha256'] for k in ('image','label')):raise ValueError('training changed source bytes')
        last=output/'runs/rectports/weights/last.pt'
        progress.update(status='complete',completed_epochs=model.trainer.epoch+1,last_checkpoint=str(last),last_sha256=sha(last),seconds=round(time.monotonic()-started,2),
            warning='Gradient/loader diagnostics only, not evaluated source accuracy or deployed production weights.')
        save(output/'report.json',dict(**progress,pins=pins,runtime=frozen,mainline_dirty_state_unchanged=True));save(output/'progress.json',progress)
        print({k:progress[k] for k in ('status','mode','completed_epochs','nonzero_gradient_steps','changed_early_backbone_tensors','changed_frozen_tensors','seconds')},flush=True)
    except BaseException as exc:
        progress.update(status='failed',error=type(exc).__name__+': '+str(exc),partial_training_not_accuracy_acceptance=True)
        save(output/'progress.json',progress);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('smoke','full'));main(parser.parse_args().mode)
