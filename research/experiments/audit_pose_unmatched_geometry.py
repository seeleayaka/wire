"""Post-selection GT diagnostics only, never runtime selection logic."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.port_tiling import box_iou
SOURCE=ROOT/'artifacts/paired_pose_search_20261003'
OUT=ROOT/'artifacts/paired_pose_unmatched_diagnostics_20261004'


def main():
    if OUT.exists():raise FileExistsError('Preserve diagnostics')
    pins={str(Path(__file__)):sha(Path(__file__))};rows=[]
    entries={r['image']:r for r in load(BASE/'train/report.json')['cases']}
    for row in load(SOURCE/'train/report.json')['cases']:
        if not row['additions']:continue
        name=row['image'];path=SOURCE/'train'/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
        targets=read_targets('train',name,[2736,3648],entries[name]['label_sha256'],pins)
        for proposal in case['trial']['paired_semantic_additions']:
            overlap=sorted([(box_iou(proposal['box_xyxy'],t['box']),i,t['class_id'],t['box']) for i,t in enumerate(targets)],reverse=True)
            same=[item for item in overlap if item[2]==proposal['class_id']]
            old=sorted([(box_iou(proposal['box_xyxy'],p['box_xyxy']),i,p['class_id'],p['box_xyxy']) for i,p in enumerate(case['current']['all_predictions'])],reverse=True)
            rows.append(dict(image=name,proposal=proposal,best_same_class_GT=same[:1],best_any_GT=overlap[:1],best_old_prediction=old[:1],
                GT_used_only_after_frozen_selection=True))
    assert {p:sha(Path(p)) for p in pins}==pins
    OUT.mkdir();save(OUT/'report.json',dict(status='complete',cases=rows,pins=pins,no_selection_change=True,field_accuracy=False))
    for row in rows:print(str({k:row[k] for k in ('image','best_same_class_GT','best_old_prediction')}),flush=True)


if __name__=='__main__':main()
