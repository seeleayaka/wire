"""TRAIN-only miss taxonomy after accepted native enhancement; no tuning."""
import sys
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import matches,metric,overlap
SOURCE=ROOT/'artifacts/paired_pose_native_three_20261004/full_train'
OUT=ROOT/'artifacts/native_remaining_train_audit_20261004'

def main():
    if OUT.exists():raise FileExistsError('Preserve completed miss audit')
    OUT.mkdir();pins={str(Path(__file__)):sha(Path(__file__))};rows=[];totals=Counter();metrics=Counter()
    for entry in load(BASE/'train/report.json')['cases']:
        name=entry['image'];path=SOURCE/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
        targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
        predictions=case['trial']['all_predictions'];used=matches(predictions,targets)[0];metrics.update(metric(predictions,targets))
        proposals=case['proposals'];scores=case['probabilities'];assert len(proposals)==len(scores)
        misses=[]
        for ti,target in enumerate(targets):
            if ti in used:continue
            compatible=[(overlap(p['box_xyxy'],target['box']),i) for i,p in enumerate(proposals) if p['class_id']==target['class_id']]
            eligible=[i for value,i in compatible if value>=.5]
            gate=[i for i in eligible if scores[i][target['class_id']+1]>=.98]
            votes=[i for i in gate if len(set(proposals[i].get('semantic_model_vote_sha256',[])))>=3]
            if not eligible:reason='no_IoU50_same_class_native_proposal'
            elif not gate:reason='classifier_probability_below_fixed_p98'
            elif not votes:reason='fewer_than_three_unique_checkpoints'
            elif len(predictions)>=10:reason='shared_5_plus_5_budget_or_selection'
            else:reason='dedup_parent_or_selection_requires_geometry_review'
            totals[reason]+=1
            best=max(compatible,default=(0.,None))
            misses.append(dict(target_index=ti,class_id=target['class_id'],box=target['box'],reason=reason,
                best_same_class_proposal_iou=best[0],eligible_proposals=eligible,
                highest_positive_probability=max((scores[i][target['class_id']+1] for i in eligible),default=None),
                accepted_cue_count=len(predictions)))
        rows.append(dict(image=name,misses=misses))
    assert metrics['tp']==295 and metrics['targets']==344 and metrics['unmatched']==4
    assert sum(totals.values())==49
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    result=dict(status='complete',train_only=True,no_validation_labels_read=True,no_training=True,
        cached_source_before_reference_runtime_filter=True,metrics=dict(metrics),taxonomy=dict(totals),cases=rows,pins=pins,
        weak_labels_not_verified_physical_faults=True,field_accuracy=False)
    save(OUT/'report.json',result);print(dict(status='complete',metrics=dict(metrics),taxonomy=dict(totals)))

if __name__=='__main__':main()
