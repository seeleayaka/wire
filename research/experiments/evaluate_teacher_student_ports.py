"""Frozen, cached model-complement study; training decides rule, heldouts score it."""
import argparse,copy,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha
from teacher_student_port_policy import merge,POLICY
from audit_port_multiscale_acceptance import matches,metric,overlap
from core_port_recheck_policy import select_zoom
OUT=ROOT/'artifacts/teacher_student_port_20261003';STUDENT=ROOT/'artifacts/port_training_multiscale_20261002'
OLD=ROOT/'artifacts/core_port_recheck_20261002'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','inner','outer'],required=True);args=parser.parse_args()
    mode=args.mode;protocol=load(OUT/'protocol.json');output=OUT/mode
    if output.exists():raise FileExistsError('Keep earlier experiment untouched')
    new_report=load(STUDENT/'evaluation'/mode/'report.json');old_report=load(OLD/mode/'report.json')
    assert new_report['status']=='complete' and not new_report['baseline_replay_only']
    full=load(STUDENT/'full/report.json');teacher_weight=REPO/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
    assert sha(teacher_weight)==protocol['teacher_sha256']
    assert sha(Path(full['weights']['last.pt']['path']))==protocol['student_sha256']
    frozen=load(STUDENT/'protocol.json');assert set(frozen['train_sources']).isdisjoint(frozen['inner_val_sources'])
    modes=POLICY['variants'] if mode=='train' else [load(OUT/'training_decision.json')['mode']]
    if mode!='train':
        decision=load(OUT/'training_decision.json');assert decision['accepted']
        assert decision['policy_sha256']==sha(Path(__file__).with_name('teacher_student_port_policy.py'))
        assert decision['protocol_sha256']==sha(OUT/'protocol.json')
        assert decision['evaluator_sha256']==sha(Path(__file__))
    input_pins={};cases=[];source_split='val01' if mode=='outer' else 'train01'
    # Verify all inference identity before opening any annotations.
    for entry in new_report['cases']:
        name=entry['image'];stem=Path(name).stem
        old_path=OLD/mode/(stem+'_zoom_predictions.json');new_path=STUDENT/'evaluation'/mode/(stem+'_zoom_predictions.json')
        old=load(old_path);new=load(new_path);source=DATA/'images'/source_split/name
        assert old['image']==new['image']==name and old['source_sha256']==new['source_sha256']==sha(source)
        assert old['weight_sha256']==protocol['teacher_sha256'] and new['weight_sha256']==protocol['student_sha256']
        if mode=='train':assert name in frozen['train_sources']
        if mode=='inner':assert name in frozen['inner_val_sources']
        for p in (old_path,new_path,source):input_pins[str(p)]=sha(p)
        candidates={m:merge(old,new,m) for m in modes}
        baseline=select_zoom(old,'strict')
        for result in candidates.values():
            assert result['primary']==baseline['primary'] and result['all_predictions'][:len(baseline['all_predictions'])]==baseline['all_predictions']
        cases.append(dict(image=name,old=old,new=new,baseline=baseline,candidates=candidates))
    output.mkdir();save(output/'inputs.json',dict(pins=input_pins,policy=POLICY,modes=modes,
        prediction_selection_before_labels=True,already_seen_same_camera_sets=True,field_accuracy_claimed=False))
    stages=['baseline','student']+modes
    totals={m:{k:0 for k in ('tp','unmatched','fn','predictions','targets')} for m in stages};normal={m:0 for m in stages};rows=[];diagnosis=[]
    for case in cases:
        name=case['image'];stem=Path(name).stem;h,w=case['old']['predictions']['source_shape'];targets=[]
        label=DATA/'labels'/source_split/(stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        selected=dict(baseline=case['baseline'],student=select_zoom(case['new'],'strict'),**case['candidates'])
        metrics={stage:metric(s['all_predictions'],targets) for stage,s in selected.items()}
        for stage,values in metrics.items():
            for k,v in values.items():totals[stage][k]+=v
            if name.startswith('normal_'):normal[stage]+=values['predictions']
        if mode=='train':
            for stage in ('primary','all_predictions'):
                old_targets=matches(selected['baseline'][stage],targets)[0];new_targets=matches(selected['student'][stage],targets)[0]
                for ti in sorted(old_targets-new_targets):
                    target=targets[ti];raw=case['new']['predictions']['merged_predictions']
                    same=[p for p in raw if p['class_id']==target['class_id'] and overlap(p['box_xyxy'],target['box'])>=.5]
                    same.sort(key=lambda p:-p['confidence'])
                    if same:reason='score_below_original_gate' if same[0]['confidence']<=.5 else 'budget_or_consistency_exclusion'
                    else:reason='no_matching_retained_same_class_box'
                    diagnosis.append(dict(image=name,stage=stage,target_index=ti,reason=reason,
                        best_matching_score=same[0]['confidence'] if same else None,
                        maximum_same_class_iou=max((overlap(p['box_xyxy'],target['box']) for p in raw if p['class_id']==target['class_id']),default=0.),
                        maximum_other_class_iou=max((overlap(p['box_xyxy'],target['box']) for p in raw if p['class_id']!=target['class_id']),default=0.)))
        save(output/(stem+'_evaluation.json'),dict(image=name,metrics=metrics,selected=selected,label_sha256=sha(label)))
        rows.append(dict(image=name,metrics=metrics,additions={m:len(case['candidates'][m]['student_additions']) for m in modes}))
    assert totals['baseline']==old_report['summary']['strict']
    assert totals['student']==new_report['summary']
    qualifies={m:(totals[m]['tp']>totals['baseline']['tp'] if mode!='outer' else totals[m]['tp']>=totals['baseline']['tp'])
        and totals[m]['unmatched']<=totals['baseline']['unmatched'] and normal[m]<=normal['baseline'] for m in modes}
    assert {p:sha(Path(p)) for p in input_pins}==input_pins
    report=dict(status='complete',summary=totals,normal_cues=normal,qualifies=qualifies,cases=rows,
        teacher_primary_and_all_old_cues_preserved=True,maximum_total_cues=10,new_inference=False,
        fingerprints_verified_cache_replay=True,production_approved=False,operator_confirmation=False,
        scope='Same-camera class-matched port localization, not physical fault or continuity accuracy')
    save(output/'report.json',report)
    if mode=='train':
        save(output/'student_regression_diagnosis.json',dict(records=diagnosis,hypotheses=['score drift','budget/consistency exclusion','localization/class drift'],training_only=True))
        accepted=[m for m in modes if qualifies[m]]
        save(OUT/'training_decision.json',dict(accepted=bool(accepted),mode=accepted[0] if accepted else None,
            policy_sha256=sha(Path(__file__).with_name('teacher_student_port_policy.py')),
            protocol_sha256=sha(OUT/'protocol.json'),evaluator_sha256=sha(Path(__file__)),no_heldout_rule_selection=True))
    print(json.dumps(dict(summary=totals,normal_cues=normal,qualifies=qualifies),indent=2),flush=True)

if __name__=='__main__':main()
