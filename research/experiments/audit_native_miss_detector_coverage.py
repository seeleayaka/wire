"""TRAIN-only detector coverage behind native misses; read-only post scoring."""
from collections import Counter
from pathlib import Path
from prepare_paired_port_semantics import ROOT,load,save,sha
from current_port_baseline_audit import BASE,read_current_case
from audit_port_multiscale_acceptance import overlap
SOURCE=ROOT/'artifacts/native_remaining_train_audit_20261004/report.json'
OUT=ROOT/'artifacts/native_miss_detector_coverage_20261004'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train/index.json'

def main():
    if OUT.exists():raise FileExistsError('Preserve detector-coverage audit')
    OUT.mkdir();pins={str(p):sha(p) for p in (SOURCE,PREP,Path(__file__))}
    index={r['image']:r for r in load(PREP)['records']};entries={r['image']:r for r in load(BASE/'train/report.json')['cases']}
    rows=[];counts=Counter()
    for row in load(SOURCE)['cases']:
        if not row['misses']:continue
        name=row['image'];item=index[name];path=Path(item['path']);assert sha(path)==item['sha256'];pins[str(path)]=sha(path)
        paired=load(path);teacher,old=read_current_case('train',entries[name],pins)
        assert teacher==paired['teacher']
        models=[teacher,paired['student'],paired['feature'],old['alternative']]
        currentpath=ROOT/'artifacts/paired_pose_native_three_20261004/full_train'/(Path(name).stem+'_predictions.json');pins[str(currentpath)]=sha(currentpath)
        current=load(currentpath)['trial']['all_predictions']
        for miss in row['misses']:
            raw=[]
            for model in models:
                best=max((overlap(p['box_xyxy'],miss['box']) for p in model['predictions']['merged_predictions']
                          if p['class_id']==miss['class_id'] and p['confidence']>.05),default=0.)
                raw.append(dict(weight_sha256=model['weight_sha256'],best_iou=best))
            existing=max((overlap(p['box_xyxy'],miss['box']) for p in current if p['class_id']==miss['class_id']),default=0.)
            votes=len({r['weight_sha256'] for r in raw if r['best_iou']>=.5})
            if existing>=.3:reason='existing_cue_near_target_localization_candidate'
            elif votes>=3:reason='three_detector_coverage_but_native_filter_or_budget'
            elif max(r['best_iou'] for r in raw)>=.5:reason='detector_coverage_without_three_checkpoint_agreement'
            elif max(r['best_iou'] for r in raw)>=.3:reason='raw_detector_near_target_localization_candidate'
            else:reason='no_raw_detector_IoU30_same_class_coverage'
            counts[reason]+=1;rows.append(dict(image=name,**miss,detector_reason=reason,raw=raw,existing_best_iou=existing,unique_raw_IoU50_votes=votes))
    assert len(rows)==49 and all(sha(Path(p))==v for p,v in pins.items())
    result=dict(status='complete',train_only=True,no_validation_read=True,no_tuning=True,
        coverage_is_post_scoring_not_runtime_GT=True,taxonomy=dict(counts),cases=rows,pins=pins,field_accuracy=False)
    save(OUT/'report.json',result);print(dict(status='complete',taxonomy=dict(counts)))

if __name__=='__main__':main()
