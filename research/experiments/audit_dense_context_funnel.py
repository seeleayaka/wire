"""Explain zero-addition outcome without changing any policy gates."""
import sys
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,load,save
from inspection_agent.port_tiling import box_iou
from inspection_agent.optional_port_crop_review import sha


def main():
    folder=ROOT/'artifacts/dense_seed_context_20261003';path=folder/'funnel_audit.json'
    if path.exists():raise FileExistsError('Preserve prior audit')
    report=load(folder/'train/report.json');assert report['status']=='complete'
    reasons=Counter();examples=[];pins={str(folder/'train/report.json'):sha(folder/'train/report.json')}
    for row in report['cases']:
        p=folder/'train'/(Path(row['image']).stem+'_predictions.json');pins[str(p)]=sha(p);case=load(p)
        for entry in case['context_evidence']:
            seed=entry['seed'];views=entry['views'];same=[[p for p in view if p['class_id']==seed['class_id']] for view in views]
            best=[max((box_iou(p['box_xyxy'],seed['box_xyxy']) for p in view),default=0) for view in same]
            eligible=[[p for p in view if box_iou(p['box_xyxy'],seed['box_xyxy'])>=.5] for view in same]
            if not any(views):reason='no_strict_head_detection_either_context'
            elif not all(same):reason='same_class_missing_one_or_both_contexts'
            elif not all(eligible):reason='seed_geometry_support_missing_one_or_both_contexts'
            elif not any(box_iou(a['box_xyxy'],b['box_xyxy'])>=.5 for a in eligible[0] for b in eligible[1]):reason='cross_context_geometry_disagreement'
            else:reason='agreement_exists_but_old_overlap_or_margin_or_budget'
            reasons[reason]+=1
            examples.append(dict(image=row['image'],seed=seed,reason=reason,strict_head_counts=[len(v) for v in views],best_same_seed_iou=best))
    result=dict(status='complete',sources=192,seed_sources=27,seeds=sum(reasons.values()),reasons=dict(reasons),cases=examples,
        original_result=report['summary'],pins=pins,no_threshold_search_or_change=True,field_accuracy=False)
    save(path,result);print(str({k:v for k,v in result.items() if k not in ('cases','pins')}))


if __name__=='__main__':main()
