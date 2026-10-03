"""Visualize every above-threshold cached port cue; no annotation selection."""
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
    report=json.loads(args.report.read_text(encoding='utf-8'))
    data=Path('E:/PythonProject10/data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images/val01')
    cases=[c for c in report['cases'] if any(b['confidence']>report['threshold'] and b['valid_warp_fraction']>=.98 for b in c['aligned_predictions'])]
    args.output.mkdir(parents=True);board=Image.new('RGB',(1600,max(1,len(cases))*650),'#20252a')
    for row,case in enumerate(cases):
        source=cv2.imdecode(np.fromfile(str(data/case['image']),dtype=np.uint8),cv2.IMREAD_COLOR)
        image=cv2.warpPerspective(source,np.asarray(case['actual_homography']),(source.shape[1],source.shape[0]))
        for column in (0,1):
            panel=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(panel)
            for b in case['parents']:draw.rectangle([b[k] for k in ('left','top','right','bottom')],outline='yellow',width=4)
            if column:
                for b in case['aligned_predictions']:
                    if b['confidence']<=report['threshold'] or b['valid_warp_fraction']<.98:continue
                    accepted=any(h['box']==b for h in case['hints'])
                    draw.rectangle([b[k] for k in ('left','top','right','bottom')],outline='cyan' if accepted else 'magenta',width=7)
            for target in case['targets']:draw.rectangle(target,outline='red',width=2)
            panel.thumbnail((790,590));board.paste(panel,(column*800,row*650+55))
            draw=ImageDraw.Draw(board);draw.text((column*800+8,row*650+8),case['image']+' | '+('frozen parents' if not column else 'all fixed-threshold port cues'),fill='white')
            draw.text((column*800+8,row*650+28),'Yellow parents / Red scoring fragments / Cyan accepted / Magenta spatially rejected',fill='white')
    board.save(args.output/'hint_and_gate_comparison.png')
    (args.output/'manifest.json').write_text(json.dumps({'images':[c['image'] for c in cases],
        'selection':'every image with an above-frozen-threshold valid detector cue, not label-selected'},indent=2),encoding='utf-8')
    print([c['image'] for c in cases])


if __name__=='__main__':main()
