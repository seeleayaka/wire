"""Freeze an inner-error-derived challenger, then recheck without outer tuning."""
import argparse, json, sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/core_complete_frame_20261002'
from core_port_precision_policy import select

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['freeze','inner','outer'],required=True)
    args=parser.parse_args()
    if args.mode=='freeze':
        if OUT.exists():raise FileExistsError('Do not overwrite protocol')
        OUT.mkdir()
        save(OUT/'protocol.json',dict(mode='high_score_complete_frame',score_threshold=.5,maximum=5,
            native_frame_margin=16,margin_origin='existing artificial tile cut safety margin',
            all_four_edges=True,derived_from_inner_errors=True,outer_labels_used_for_design=False,
            accepts_no_coordinates_or_image_names=True,gate='TP unchanged and unmatched lower vs high_score',
            limitation='Incomplete genuine ports also abstain; not a physical fault verdict'))
        print('COMPLETE-FRAME POLICY FROZEN');return
    if args.mode=='outer':
        inner=load(OUT/'inner_report.json');assert inner['qualifies']
    from core_port_resolution_ab_20261002 import DATA,score,sha
    base=ROOT/('artifacts/core_precision_full_'+args.mode+'_20261002')
    previous=load(base/'report.json');split='train01' if args.mode=='inner' else 'val01'
    totals={};cases=[]
    for case in previous['cases']:
        raw=load(base/(Path(case['image']).stem+'_predictions.json'))
        assert sha(DATA/'images'/split/case['image'])==raw['source_sha256']
        selected={m:select(raw['predictions'],m) for m in ('high_score','high_score_complete_frame')}
        label=DATA/'labels'/split/(Path(case['image']).stem+'.txt');assert sha(label)==case['label_sha256']
        targets=[];height,width=raw['predictions']['source_shape']
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height]))
        metrics={m:score(rows,targets) for m,rows in selected.items()}
        for m,metric in metrics.items():
            total=totals.setdefault(m,{k:0 for k in metric})
            for k,v in metric.items():total[k]+=v
        cases.append(dict(image=case['image'],metrics=metrics))
    before=totals['high_score'];after=totals['high_score_complete_frame']
    save(OUT/(args.mode+'_report.json'),dict(summary=totals,cases=cases,
        qualifies=after['tp']==before['tp'] and after['unmatched']<before['unmatched'],
        no_regression=after['tp']==before['tp'] and after['unmatched']<=before['unmatched'],
        new_inference=False,frame_policy='fixed 16 margin, all four edges',
        already_inspected_images=True,field_accuracy_claimed=False))
    print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
