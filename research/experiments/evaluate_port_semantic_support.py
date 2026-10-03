"""Actual detector boxes, source-group OOF heads, training-only acceptance."""
import json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/port_semantic_source_support_20261003'
HEAD=ROOT/'artifacts/port_semantic_verifier_20261003'
HIGH=ROOT/'artifacts/port_high_resolution_support_20261003'
BASE=ROOT/'artifacts/port_residual_feature_support_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ['HF_HUB_OFFLINE']='1'
from inspection_agent.optional_port_crop_review import sha,read_image
from port_semantic_verifier import embeddings
from port_semantic_support_policy import proposals,append_semantic
from audit_port_multiscale_acceptance import metric,matches
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    if OUT.exists():raise FileExistsError('No repeated fitting/evaluation')
    feasibility=load(HEAD/'report.json');OUT.mkdir()
    if not feasibility['qualifies_crop_feasibility']:
        save(OUT/'progress.json',dict(status='rejected_before_detector_inference',reason='source_group_oof_crop_gate_failed'))
        return
    training=load(HEAD/'protocol.json');names=training['train_sources'];fold_for={name:index%3 for index,name in enumerate(names)}
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_semantic_verifier.py'),
        Path(__file__).with_name('port_semantic_support_policy.py'),Path(__file__).with_name('audit_port_multiscale_acceptance.py'),
        HEAD/'protocol.json',HEAD/'report.json',HEAD/'features.pt',HEAD/'samples.json')}
    for p,digest in training['pins'].items():assert sha(Path(p))==digest;pins[p]=digest
    headpaths=[HEAD/f'fold{fold}_head.pt' for fold in range(3)]
    for p in headpaths:pins[str(p)]=sha(p)
    inputs=[]
    for mode in ('train','extended'):
        for row in load(HIGH/mode/'report.json')['cases']:
            name=row['image'];stem=Path(name).stem
            highfile=HIGH/mode/(stem+'_predictions.json');featurefile=BASE/mode/(stem+'_predictions.json')
            if mode=='extended':
                oldfile=ROOT/'artifacts/port_extended_training_controls_20261003'/(stem+'_predictions.json')
                old=load(oldfile)['versions'];teacher,student=old['teacher'],old['student'];files=[oldfile]
            else:
                teacherfile=ROOT/'artifacts/core_port_recheck_20261002/train'/(stem+'_zoom_predictions.json')
                studentfile=ROOT/'artifacts/port_training_multiscale_20261002/evaluation/train'/(stem+'_zoom_predictions.json')
                teacher,student=load(teacherfile),load(studentfile);files=[teacherfile,studentfile]
            feature=load(featurefile)['feature'];high=load(highfile);source=DATA/'images/train01'/name
            for p in [*files,highfile,featurefile,source]:pins[str(p)]=sha(p)
            assert name in fold_for and teacher['source_sha256']==pins[str(source)]
            inputs.append(dict(image=name,source=source,teacher=teacher,models=[teacher,student,feature,high['alternative']],current=high['trial']))
    assert len(inputs)==168
    save(OUT/'protocol.json',dict(pins=pins,images=[r['image'] for r in inputs],
        actual_detector_box_evaluation=True,all_original32_and136_negative_controls=True,
        oof_head_by_whole_source=True,semantic_probability_gate=.98,
        new_candidate_detector_floor=.25,two_complete_distinct_raw_tiles_above_floor=True,
        teacher_sameclass_support_above_floor=True,
        original_detector_rules_and_cues_unchanged=True,new_independent_confirmation_is_not_probability_averaging=True,
        max_primary=5,max_extra=5,require_training_net_tp_gain=True,no_unmatched_increase=True,no_old_target_loss=True,
        no_validation_images=True,no_reference_checks_yet=True,production_changed=False,field_accuracy=False))
    import torch
    import dino_feature_diff as dino
    torch.set_num_threads(2);model=dino._model();model.requires_grad_(False);torch.set_num_threads(2)
    heads=[]
    for p in headpaths:
        head=torch.nn.Linear(1536,3);head.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));head.eval();heads.append(head)
    started=time.monotonic();records=[]
    try:
        for index,row in enumerate(inputs):
            name=row['image'];current=row['current'];teacher=row['teacher']
            room=len(current['all_predictions'])-len(current['primary'])<5
            candidates=proposals(teacher,row['models']) if room else []
            if candidates:
                features=embeddings(model,read_image(row['source']),[r['box_xyxy'] for r in candidates])
                with torch.inference_mode():probabilities=heads[fold_for[name]](features).softmax(dim=1).tolist()
            else:probabilities=[]
            trial=append_semantic(current,candidates,probabilities)
            save(OUT/(Path(name).stem+'_predictions.json'),dict(image=name,fold=fold_for[name],current=current,
                trial=trial,candidates=candidates,probabilities=probabilities))
            label=DATA/'labels/train01'/(Path(name).stem+'.txt');h,w=teacher['predictions']['source_shape'];targets=[]
            for line in label.read_text(encoding='utf-8').splitlines():
                cls,cx,cy,bw,bh=map(float,line.split())
                if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
            old,new=current['all_predictions'],trial['all_predictions'];oldhits,newhits=matches(old,targets)[0],matches(new,targets)[0]
            record=dict(image=name,metrics=dict(current=metric(old,targets),trial=metric(new,targets)),
                additions=len(trial['semantic_additions']),gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits),label_sha256=sha(label))
            records.append(record)
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=index+1,total=len(inputs),image=name,
                seconds=round(time.monotonic()-started,2)))
            save(OUT/'partial.json',dict(cases=records,seconds=round(time.monotonic()-started,2)))
            print(json.dumps(record),flush=True)
        totals={s:{key:sum(r['metrics'][s][key] for r in records) for key in ('tp','unmatched','fn','predictions','targets')} for s in ('current','trial')}
        assert {p:sha(Path(p)) for p in pins}==pins
        assert totals['current']['tp']==69 and totals['current']['unmatched']==4
        normal=sum(r['metrics']['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
        passed=(totals['trial']['tp']>totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and
            not any(r['lost'] for r in records) and normal==0)
        save(OUT/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,
            seconds=round(time.monotonic()-started,2),reference_pending=True,validation_not_run=True,
            field_accuracy=False,production_changed=False))
        save(OUT/'progress.json',dict(status='complete',qualifies=passed,summary=totals))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise
if __name__=='__main__':main()
