"""Diagnostic source-pixel crops only; no annotation changes or rule selection."""
import argparse, json, sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
import cv2
from inspection_agent.optional_port_crop_review import read_image,sha
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/core_precision_full_inner_20261002'
OUT=BASE/'remaining_review'
DATA=Path('E:/PythonProject10/data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outer',action='store_true');args=parser.parse_args()
    base=ROOT/'artifacts/core_precision_full_outer_20261002' if args.outer else BASE
    out=base/'remaining_review'
    report_path=ROOT/'artifacts/core_complete_frame_20261002/outer_report.json' if args.outer else base/'report.json'
    stage='high_score_complete_frame' if args.outer else 'high_score'
    split='val01' if args.outer else 'train01'
    if out.exists():raise FileExistsError('Fresh output required')
    out.mkdir()
    report=json.loads(report_path.read_text(encoding='utf-8'))
    rows=[]
    for case in report['cases']:
        if not case['metrics'][stage]['unmatched']:continue
        path=DATA/'images'/split/case['image'];image=read_image(path)
        if args.outer:
            from core_port_precision_policy import select
            raw=json.loads((base/(Path(case['image']).stem+'_predictions.json')).read_text())
            selected=select(raw['predictions'],stage)
        else:
            selected=json.loads((base/(Path(case['image']).stem+'_evaluation.json')).read_text())['selected'][stage]
        for index,prediction in enumerate(selected):
            l,t,r,b=map(round,prediction['box_xyxy']);height,width=image.shape[:2]
            x,y=max(0,l-200),max(0,t-180)
            right,bottom=min(width,r+200),min(height,b+180)
            crop=image[y:bottom,x:right].copy()
            cv2.rectangle(crop,(l-x,t-y),(r-x,b-y),(0,0,255),2)
            target=out/(path.stem+f'_{index}.jpg')
            assert cv2.imwrite(str(target),crop)
            rows.append(dict(image=path.name,source_sha256=sha(path),box=prediction,
                source_crop_xyxy=[x,y,right,bottom],review_image=str(target),
                operator_confirmed=False,rule_changed=False))
    (out/'review.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
