"""Post-hoc grid-evidence diagnostic; source box overlap is not pixel truth."""
import argparse
import json
from pathlib import Path
import numpy as np


def target_grid(targets, bounds, grid):
    left, top, right, bottom = bounds
    height, width = grid
    mask = np.zeros(grid, dtype=bool)
    for x0,y0,x1,y1 in targets:
        x0,x1 = max(x0,left),min(x1,right)
        y0,y1 = max(y0,top),min(y1,bottom)
        if x0>=x1 or y0>=y1:
            continue
        xa,xb = int(np.floor((x0-left)*width/(right-left))),int(np.ceil((x1-left)*width/(right-left)))
        ya,yb = int(np.floor((y0-top)*height/(bottom-top))),int(np.ceil((y1-top)*height/(bottom-top)))
        mask[max(0,ya):min(height,yb),max(0,xa):min(width,xb)] = True
    return mask


def rank_auc(score, mask):
    positive, negative = score[mask], score[~mask]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean((positive[:,None]>negative[None,:]).astype(float)
                         +0.5*(positive[:,None]==negative[None,:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps',type=Path,required=True)
    parser.add_argument('--fault-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rows=[]
    for path in sorted(args.fault_cache.glob('*.json')):
        if path.name=='comparison.json':
            continue
        record=json.loads(path.read_text(encoding='utf-8'))
        assert record['split']=='val01' and not record['image'].startswith('normal_')
        with np.load(args.maps/(Path(record['image']).stem+'_maps.npz'),allow_pickle=False) as data:
            mask=target_grid(record['targets'],record['bounds'],data['local'].shape)
            row={'image':record['image'],'marked_grid_cells':int(mask.sum()),'total_grid_cells':int(mask.size)}
            for branch in ('local','position','fusion'):
                score=data[branch]
                row[branch]={'grid_overlap_rank_auc':rank_auc(score,mask),
                             'marked_median':float(np.median(score[mask])) if mask.any() else None,
                             'unmarked_p95':float(np.percentile(score[~mask],95)) if (~mask).any() else None}
            rows.append(row)
    assert len(rows)==15
    summaries={}
    for kind in ('all','damaged','disconnected','misrouted'):
        selected=rows if kind=='all' else [r for r in rows if r['image'].startswith(kind+'_')]
        summaries[kind]={branch:float(np.mean([r[branch]['grid_overlap_rank_auc'] for r in selected
                                              if r[branch]['grid_overlap_rank_auc'] is not None]))
                         for branch in ('local','position','fusion')}
    result={'posthoc':True,'parameters_changed':False,'image_equal_mean_grid_auc':summaries,'cases':rows,
            'warning':'Coarse grid intersects fragmented source boxes, not pixel-level defect truth. Unmarked areas may contain faults. This diagnostic is not a fault classifier metric or proof of cause.'}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
