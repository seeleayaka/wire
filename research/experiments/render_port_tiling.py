"""First-kind controls plus above-threshold predictions; never select by target IoU."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    report=json.loads(args.report.read_text(encoding='utf-8'));assert report['status']=='complete'
    cases=report['cases'];selected={min(c['image'] for c in cases if c['kind']==kind)
                                 for kind in ('damaged','disconnected','misrouted','normal')}
    selected|={c['image'] for c in cases if c['tile_hints']}
    selected|={c['image'] for c in cases if any(p['confidence']>report['threshold'] for p in c['aligned_predictions'])}
    names=sorted(selected);args.output.mkdir(parents=True)
    data=Path('E:/PythonProject10/data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images')
    reference=cv2.imdecode(np.fromfile(str(data/'train01/normal_073.JPG'),dtype=np.uint8),cv2.IMREAD_COLOR)
    board=Image.new('RGB',(1600,650*len(names)),'#20252a')
    for row,name in enumerate(names):
        case=next(c for c in cases if c['image']==name)
        source=cv2.imdecode(np.fromfile(str(data/'val01'/name),dtype=np.uint8),cv2.IMREAD_COLOR)
        warped=cv2.warpPerspective(source,np.asarray(case['actual_homography']),(reference.shape[1],reference.shape[0]))
        for col in (0,1):
            panel=Image.fromarray(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(panel)
            for b in case['parents']:draw.rectangle([b[k] for k in ('left','top','right','bottom')],outline='yellow',width=4)
            for h in case['existing_hints']:draw.rectangle([h['box'][k] for k in ('left','top','right','bottom')],outline='cyan',width=6)
            if col:
                for prediction in case['aligned_predictions']:
                    if prediction['confidence']>report['threshold']:
                        draw.rectangle([prediction[k] for k in ('left','top','right','bottom')],outline='orange',width=5)
                for h in case['tile_hints']:draw.rectangle([h['box'][k] for k in ('left','top','right','bottom')],outline='magenta',width=6)
            for target in case['targets']:draw.rectangle(target,outline='red',width=2)
            panel.thumbnail((790,585));cell=Image.new('RGB',(800,650),'#20252a');cell.paste(panel,(0,55))
            label=ImageDraw.Draw(cell);label.text((8,8),name+' | '+('previous full-frame hint' if not col else 'additional tiled hint'),fill='white')
            label.text((8,28),'Yellow parent / Cyan old / Orange gated prediction / Magenta accepted / Red GT',fill='white')
            cell.save(args.output/(Path(name).stem+('_old.png' if not col else '_tiled.png')))
            board.paste(cell,(col*800,row*650))
    board.save(args.output/'comparison.png')
    (args.output/'manifest.json').write_text(json.dumps({'images':names,
        'selection':'first filename per kind plus all added-hint/above-threshold images, not target-IoU selected'},indent=2),encoding='utf-8')
    print(names)


if __name__=='__main__':main()
