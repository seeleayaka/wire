"""Compare annotation geometry with the frozen feature grid, without inference."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.cache.glob('*.json')) if p.name!='comparison.json']
    assert len(records)==15 and all(r['split']=='val01' for r in records)
    result={'split':'val01','grid_hw':[21,28],'by_kind':{}}
    for kind in ('all','damaged','disconnected','misrouted'):
        selected=records if kind=='all' else [r for r in records if r['image'].startswith(kind+'_')]
        rows=[]
        for r in selected:
            left,top,right,bottom=r['bounds']
            patch_w=(right-left)/28; patch_h=(bottom-top)/21
            for x0,y0,x1,y1 in r['targets']:
                rows.append([(x1-x0),(y1-y0),(x1-x0)/patch_w,(y1-y0)/patch_h])
        a=np.asarray(rows)
        result['by_kind'][kind]={'targets':len(rows),'median_width_px':float(np.median(a[:,0])),
            'median_height_px':float(np.median(a[:,1])),
            'both_dimensions_below_one_patch':int(np.sum((a[:,2]<1)&(a[:,3]<1))),
            'both_dimensions_below_half_patch':int(np.sum((a[:,2]<.5)&(a[:,3]<.5))),
            'median_patch_width':float(np.median(a[:,2])), 'median_patch_height':float(np.median(a[:,3]))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
