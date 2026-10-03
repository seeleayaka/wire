"""Training-control miss causes only; never select inference boxes from labels."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
from core_port_precision_policy import iou
ROOT=Path(__file__).resolve().parents[1]/'artifacts/core_port_supplement_20261002/train'

def main():
    path=ROOT/'model_miss_diagnosis.json'
    if path.exists():raise FileExistsError('Fresh diagnostic output required')
    report=json.loads((ROOT/'report.json').read_text(encoding='utf-8'));rows=[];counts={}
    for miss in report['misses']:
        if miss['cause']!='no_precise_same_class_prediction_above_0.25':continue
        raw=json.loads((ROOT/(Path(miss['image']).stem+'_predictions.json')).read_text(encoding='utf-8'))['predictions']['merged_predictions']
        same=[p for p in raw if p['class_id']==miss['class_id'] and iou(miss['target_box'],p['box_xyxy'])>=.5]
        any_class=[p for p in raw if iou(miss['target_box'],p['box_xyxy'])>=.5]
        cause='precise_but_score_below_0.25' if same else 'wrong_class_precise_box' if any_class else 'no_precise_box_in_kept_predictions'
        counts[cause]=counts.get(cause,0)+1
        rows.append(dict(**miss,detailed_cause=cause,maximum_precise_score=max((p['confidence'] for p in same),default=None)))
    path.write_text(json.dumps(dict(training_controls_only=True,counts=counts,rows=rows,
        no_parameter_selection=True,no_outer_or_test_reads=True),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
