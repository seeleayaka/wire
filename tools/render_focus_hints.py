"""Fixed first-image-per-kind visual audit; hints cyan, parents yellow, labels red."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('E:/PythonProject10')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'prototype'))
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
    if args.output.exists():raise FileExistsError('fresh directory required')
    report=json.loads(args.report.read_text(encoding='utf-8'));assert report['split']=='val01'
    dataset=ROOT/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
    reference=read(dataset/'train01/normal_073.JPG');args.output.mkdir(parents=True)
    board=Image.new('RGB',(1400,1050),'#20252a');names=[];alignments=[]
    for index,kind in enumerate(('damaged','disconnected','misrouted','normal')):
        case=min((r for r in report['cases'] if r['image'].startswith(kind+'_')),key=lambda r:r['image'])
        cv2.setRNGSeed(20260929)
        aligned,info=perspective.automatic_homography(reference,read(dataset/'val01'/case['image']))
        if aligned is None:raise ValueError('visual registration failed')
        panel=Image.fromarray(cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(panel)
        for target in case['targets']:draw.rectangle(target,outline='red',width=2)
        for index_parent,parent in enumerate(case['candidates']):
            values=[parent[k] for k in ('left','top','right','bottom')]
            draw.rectangle(values,outline='yellow',width=4)
            draw.text((values[0],values[1]),'P'+str(index_parent+1),fill='yellow')
        for hint in case['hints']:draw.rectangle([hint[k] for k in ('left','top','right','bottom')],outline='cyan',width=5)
        panel.thumbnail((690,465));cell=Image.new('RGB',(700,525),'#20252a');cell.paste(panel,(0,55))
        label=ImageDraw.Draw(cell)
        label.text((10,8),case['image']+' | parents='+str(len(case['candidates']))+' hints='+str(len(case['hints'])),fill='white')
        label.text((10,27),'Yellow: parent | Cyan: hint | Red: source fragments (scoring only)',fill='white')
        cell.save(args.output/(Path(case['image']).stem+'.png'));board.paste(cell,((index%2)*700,(index//2)*525))
        names.append(case['image']);alignments.append({'image':case['image'],'visual_alignment':info})
    board.save(args.output/'fixed_four.png')
    (args.output/'manifest.json').write_text(json.dumps({'names':names,'selection':'fixed smallest filename per kind','alignments':alignments,'warning':'Fresh registration for visual inspection only; does not recompute or change frozen metrics.'},indent=2),encoding='utf-8')
    print(names)


if __name__=='__main__':main()
