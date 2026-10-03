"""Unchanged-threshold cached ablation: model misses vs spatial guard losses."""
import argparse
import json
from pathlib import Path


def iou(a,b):
    inter=max(0,min(a['right'],b[2])-max(a['left'],b[0]))*max(0,min(a['bottom'],b[3])-max(a['top'],b[1]))
    area=(a['right']-a['left'])*(a['bottom']-a['top'])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area>0 else 0.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    r=json.loads(args.report.read_text(encoding='utf-8'));assert r['status']=='complete' and r['split']=='val01'
    rows=[];totals={n:{'fault_regions':0,'normal_regions':0,'precise_fragments':0,'overlap_fragments':0,'class_matched_precise_fragments':0}
                    for n in ('parents','within_parent_hints','all_frozen_threshold_port_boxes','parents_plus_all_port_boxes','parents_plus_top1_port_box')}
    for case in r['cases']:
        predictions=[b for b in case['aligned_predictions'] if b['confidence']>r['threshold'] and b['valid_warp_fraction']>=.98]
        top1=sorted(predictions,key=lambda b:(-b['confidence'],b['left'],b['top'],b['right'],b['bottom'],b['class_id']))[:1]
        groups={'parents':case['parents'],'within_parent_hints':[h['box'] for h in case['hints']],
                'all_frozen_threshold_port_boxes':predictions,'parents_plus_all_port_boxes':case['parents']+predictions,
                'parents_plus_top1_port_box':case['parents']+top1}
        metrics={}
        for name,boxes in groups.items():
            best=[max((iou(b,t) for b in boxes),default=0.) for t in case['targets']]
            strict=sum(any(iou(b,t)>=.5 and b.get('class_id',-10)+3==cls for b in boxes)
                       for t,cls in zip(case['targets'],case['source_classes']))
            metrics[name]={'regions':len(boxes),'precise_fragments':sum(v>=.5 for v in best),
                           'overlap_fragments':sum(v>0 for v in best),'class_matched_precise_fragments':strict}
            total=totals[name];total['fault_regions' if case['targets'] else 'normal_regions']+=len(boxes)
            for field in ('precise_fragments','overlap_fragments','class_matched_precise_fragments'):total[field]+=metrics[name][field]
        rows.append({'image':case['image'],'above_frozen_threshold_valid_boxes':len(predictions),
                     'selection_audit':case['selection_audit'],'metrics':metrics})
    result={'split':'val01','threshold_unchanged':r['threshold'],'totals':totals,'per_image':rows,
        'warning':'Retrospective cached diagnostic ablations, not new independent validation or approved integration. Top1 is confidence-only diagnostic; no annotation-based choice. No threshold adjustment or inference.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
