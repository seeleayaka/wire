"""Fixed val first-name panels; comparison only, no parameter changes."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('E:/PythonProject10');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'prototype'))
from tools.merge_audit import setup
setup(ROOT)
import assembly_auto_review_robust_v3 as perspective


def read(path):
    result=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if result is None:raise ValueError('image missing')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('use fresh output')
    report=json.loads(args.report.read_text(encoding='utf-8'));assert report['status']=='complete' and report['split']=='val01'
    dataset=ROOT/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
    reference=read(dataset/'train01/normal_073.JPG');args.output.mkdir(parents=True)
    board=Image.new('RGB',(1400,2100),'#20252a');names=[]
    for row,kind in enumerate(('damaged','disconnected','misrouted','normal')):
        case=min((c for c in report['cases'] if c['image'].startswith(kind+'_')),key=lambda c:c['image'])
        cv2.setRNGSeed(0);aligned,meta=perspective.automatic_homography(reference,read(dataset/'val01'/case['image']))
        if aligned is None:raise ValueError('visual alignment failed')
        for column,mode in enumerate(('parents','fine_selected')):
            panel=Image.fromarray(cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(panel)
            for target in case['targets']:draw.rectangle(target,outline='red',width=2)
            for parent in case[mode]:draw.rectangle([parent[k] for k in ('left','top','right','bottom')],outline='yellow' if column==0 else 'cyan',width=5)
            panel.thumbnail((690,465));cell=Image.new('RGB',(700,525),'#20252a');cell.paste(panel,(0,55))
            label=ImageDraw.Draw(cell);label.text((10,8),case['image']+' | '+mode+' | '+str(len(case[mode]))+' regions',fill='white')
            label.text((10,27),'Red: scoring-only fragments | '+('Yellow: frozen coarse parents' if column==0 else 'Cyan: fine ranking'),fill='white')
            cell.save(args.output/(Path(case['image']).stem+'_'+mode+'.png'));board.paste(cell,(column*700,row*525))
        names.append(case['image'])
    board.save(args.output/'fixed_comparison.png')
    (args.output/'manifest.json').write_text(json.dumps({'names':names,'selection':'smallest filename per kind','warning':'Fresh visual registration only, frozen metrics unchanged.'},indent=2),encoding='utf-8')
    print(names)


if __name__=='__main__':main()
