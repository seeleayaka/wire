"""Training ALL192 first, then reused validation, preserve every old cue."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
HIGH=ROOT/'artifacts/port_high_resolution_support_20261003';EXTRA=ROOT/'artifacts/remaining_training_ports_20261003'
OUT=ROOT/'artifacts/port_resolution_loose_plug_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.feature_residual_port_support import residual_runtime_fingerprint
from port_resolution_plug_policy import append_resolution_plugs,POLICY_ID
from audit_port_multiscale_acceptance import metric,matches
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Preserve experiment')
    assert load(EXTRA/'report.json')['status']=='complete'
    frozen=residual_runtime_fingerprint(REPO)
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_resolution_plug_policy.py'),
        Path(__file__).with_name('port_resolution_support.py'),Path(__file__).with_name('teacher_student_port_policy.py'),
        Path(__file__).with_name('audit_port_multiscale_acceptance.py'))}
    stages=dict(train=[HIGH/'train',HIGH/'extended',EXTRA],inner=[HIGH/'inner'],outer=[HIGH/'outer'])
    inputs={}
    for stage,folders in stages.items():
        inputs[stage]=[]
        for folder in folders:
            for row in load(folder/'report.json')['cases']:
                name=row['image'];stem=Path(name).stem;p=folder/(stem+'_predictions.json');record=load(p)
                if folder==EXTRA:teacher=record['teacher'];teacherfile=p
                elif folder.name=='extended':
                    teacherfile=ROOT/'artifacts/port_extended_training_controls_20261003'/(stem+'_predictions.json')
                    teacher=load(teacherfile)['versions']['teacher']
                else:
                    teacherfile=ROOT/'artifacts/core_port_recheck_20261002'/stage/(stem+'_zoom_predictions.json');teacher=load(teacherfile)
                source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
                for path in (p,teacherfile,source):pins[str(path)]=sha(path)
                assert teacher['source_sha256']==record['alternative']['source_sha256']==pins[str(source)]
                inputs[stage].append(dict(image=name,teacher=teacher,current=record['current'],alternative=record['alternative'],label_sha256=row['label_sha256']))
    assert len(inputs['train'])==192 and len(inputs['inner'])==48 and len(inputs['outer'])==30
    OUT.mkdir();save(OUT/'protocol.json',dict(policy_id=POLICY_ID,pins=pins,runtime_fingerprint=frozen,
        rationale='Training new-resolution candidates: loose-plug3 gains/0 unmatched; empty-jack1 gain/4 unmatched. Restrict new channel by class, preserve all old jack/plug cues',
        source_prediction_cached_sha_verified=True,only_class0_loose_plug_additions=True,
        no_semantic_head_or_low_score_candidates=True,no_old_threshold_change=True,max_primary=5,max_extra=5,
        all192_training_then_inner48_then_outer30=True,
        reused_validation_already_seen_including_candidate_classes=True,not_new_independent_validation=True,
        net_train_gain_require_no_unmatched_increase_no_old_target_loss=True,
        inner_outer_nonregression=True,reference_pending=True,production_changed=False,field_accuracy=False))
    for stage,cases in inputs.items():
        folder=OUT/stage;folder.mkdir();records=[]
        for row in cases:
            result=append_resolution_plugs(row['teacher'],row['current'],row['alternative'])
            assert result['resolution_fallback_reason'] is None
            name=row['image'];stem=Path(name).stem;h,w=row['alternative']['predictions']['source_shape'];targets=[]
            label=DATA/'labels'/('val01' if stage=='outer' else 'train01')/(stem+'.txt');assert sha(label)==row['label_sha256']
            for line in label.read_text(encoding='utf-8').splitlines():
                cls,cx,cy,bw,bh=map(float,line.split())
                if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
            old,new=row['current']['all_predictions'],result['all_predictions'];oldhits,newhits=matches(old,targets)[0],matches(new,targets)[0]
            records.append(dict(image=name,metrics=dict(current=metric(old,targets),trial=metric(new,targets)),
                gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits),additions=len(result['resolution_additions']),label_sha256=sha(label)))
            save(folder/(stem+'_predictions.json'),dict(image=name,current=row['current'],trial=result,alternative=row['alternative']))
        totals={s:{k:sum(r['metrics'][s][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for s in ('current','trial')}
        gains=totals['trial']['tp']>totals['current']['tp'] if stage=='train' else totals['trial']['tp']>=totals['current']['tp']
        normal=sum(r['metrics']['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
        passed=gains and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
        assert {p:sha(Path(p)) for p in pins}==pins and residual_runtime_fingerprint(REPO)==frozen
        save(folder/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,
            cached_replay=True,reference_pending=True,field_accuracy=False,production_changed=False))
        print(json.dumps(dict(stage=stage,qualifies=passed,summary=totals)),flush=True)
        if not passed:save(OUT/'progress.json',dict(status='rejected',stage=stage));return
    save(OUT/'progress.json',dict(status='source_only_pass_requires_live_reference',production_changed=False))
if __name__=='__main__':main()
