"""Frozen hotspot/normal-background diagnostics; no detector retuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np


def top_mask(score, fraction=0.05):
    score=np.asarray(score)
    if score.ndim!=2 or not np.isfinite(score).all() or not 0<fraction<=1:
        raise ValueError('invalid score map')
    count=max(1,int(np.ceil(score.size*fraction)))
    indices=np.argsort(-score.ravel(),kind='stable')[:count]
    mask=np.zeros(score.size,dtype=bool); mask[indices]=True
    return mask.reshape(score.shape)


def overlap_fraction(hot, region):
    if hot.shape!=region.shape or not hot.any():
        raise ValueError('invalid diagnostic masks')
    return float(np.mean(region[hot]))


def average_rank(value):
    flat=np.asarray(value,dtype=float).ravel()
    order=np.argsort(flat,kind='stable'); ranked=np.empty(len(flat),dtype=float)
    start=0
    while start<len(flat):
        end=start+1
        while end<len(flat) and flat[order[start]]==flat[order[end]]:
            end+=1
        ranked[order[start:end]]=(start+end-1)/2
        start=end
    return ranked


def rank_correlation(first,second):
    if np.shape(first)!=np.shape(second):
        raise ValueError('correlation geometry mismatch')
    a,b=average_rank(first),average_rank(second)
    if a.std()==0 or b.std()==0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def panel(image,hot,targets,bounds,label):
    left,top,right,bottom=bounds
    crop=image[top:bottom,left:right]
    height=340; width=int(round(crop.shape[1]*height/crop.shape[0]))
    frame=cv2.resize(crop,(width,height))
    if hot is not None:
        overlay=frame.copy(); rows,cols=hot.shape
        for y,x in zip(*np.where(hot)):
            cv2.rectangle(overlay,(int(x*width/cols),int(y*height/rows)),
                          (int((x+1)*width/cols)-1,int((y+1)*height/rows)-1),(0,0,255),-1)
        frame=cv2.addWeighted(frame,0.55,overlay,0.45,0)
    for x0,y0,x1,y1 in targets:
        cv2.rectangle(frame,(int((x0-left)*width/(right-left)),int((y0-top)*height/(bottom-top))),
                      (int((x1-left)*width/(right-left)),int((y1-top)*height/(bottom-top))),
                      (255,255,0),1)
    result=np.zeros((height+35,width,3),dtype=np.uint8)
    result[35:]=frame
    cv2.putText(result,label,(8,23),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    parser.add_argument('--maps',type=Path,required=True)
    parser.add_argument('--fault-cache',type=Path,required=True)
    parser.add_argument('--normal-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new diagnostics directory')
    sys.path.insert(0,str(args.repo)); sys.path.insert(0,str(args.repo/'prototype'))
    from tools.audit_spatial_evidence import target_grid
    from evaluate_mendeley_balanced import read_image
    import assembly_auto_review_robust_v3 as perspective
    source=json.loads((args.maps/'report.json').read_text(encoding='utf-8'))
    records=[]
    for directory in (args.fault_cache,args.normal_cache):
        records += [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('*.json'))
                    if p.name!='comparison.json']
    assert len(records)==30 and all(r['split']=='val01' for r in records)
    bounds=source['bounds']; grid=tuple(source['grid'][:2])
    assert all(r['bounds']==bounds for r in records)
    with np.load(args.maps/'calibration_maps.npz',allow_pickle=False) as data:
        normal_median={branch:np.median(data[branch],axis=0) for branch in ('local','position','fusion')}
        normal_hot={branch:top_mask(normal_median[branch]) for branch in normal_median}
    border=np.zeros(grid,dtype=bool); border[[0,-1],:]=True; border[:,[0,-1]]=True
    dataset=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    reference=read_image(dataset/'images/train01/normal_073.JPG')
    left,top,right,bottom=bounds
    reference_gray=cv2.cvtColor(reference[top:bottom,left:right],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    dx=cv2.Sobel(reference_gray,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(reference_gray,cv2.CV_32F,0,1,ksize=3)
    edge_grid=cv2.resize(np.hypot(dx,dy),(grid[1],grid[0]),interpolation=cv2.INTER_AREA)
    edge_hot=top_mask(edge_grid)
    args.output.mkdir(parents=True)
    rows=[]; selected={}
    for kind in ('damaged','disconnected','misrouted','normal'):
        selected[kind]=min(r['image'] for r in records if r['image'].startswith(kind+'_'))
    illustrations=[]
    for index,record in enumerate(records,1):
        name=record['image']
        with np.load(args.maps/(Path(name).stem+'_maps.npz'),allow_pickle=False) as data:
            maps={branch:data[branch] for branch in ('local','position','fusion')}
        target=target_grid(record['targets'],bounds,grid)
        row={'image':name,'target_grid_fraction':float(target.mean())}
        for branch,score in maps.items():
            hot=top_mask(score)
            row[branch]={'hot_target_overlap':overlap_fraction(hot,target),
                         'hot_normal_template_overlap':overlap_fraction(hot,normal_hot[branch]),
                         'hot_border_overlap':overlap_fraction(hot,border),
                         'hot_reference_edge_overlap':overlap_fraction(hot,edge_hot),
                         'normal_template_rank_correlation':rank_correlation(score,normal_median[branch]),
                         'p50_over_p95':float(np.percentile(score,50)/max(float(np.percentile(score,95)),1e-8))}
        # A fresh registration is diagnostic only; do not replace the original
        # feature alignment or transformed source annotations with this result.
        cv2.setRNGSeed(0)
        aligned,alignment=perspective.automatic_homography(reference,read_image(dataset/'images/val01'/name))
        row['fresh_registration']=alignment
        row['fresh_registration_not_original_cached_transform']=True
        if aligned is not None:
            valid=perspective.auto.LAST_WARP_VALID_MASK[top:bottom,left:right]>0
            gray=cv2.cvtColor(aligned[top:bottom,left:right],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            residual=np.abs(gray-reference_gray)
            numerator=cv2.resize(residual*valid,(grid[1],grid[0]),interpolation=cv2.INTER_AREA)
            coverage=cv2.resize(valid.astype(np.float32),(grid[1],grid[0]),interpolation=cv2.INTER_AREA)
            residual_grid=numerator/np.maximum(coverage,1e-6)
            valid_cells=coverage>=0.99
            row['valid_grid_fraction']=float(valid_cells.mean())
            for branch,score in maps.items():
                row[branch]['pixel_residual_rank_correlation']=rank_correlation(score[valid_cells],residual_grid[valid_cells])
                row[branch]['hot_invalid_warp_fraction']=overlap_fraction(top_mask(score),~valid_cells)
            if name in selected.values():
                panels=[panel(aligned,None,record['targets'],bounds,name+' aligned; cyan=source boxes')]
                panels += [panel(aligned,top_mask(maps[b]),record['targets'],bounds,b+' top 5% cells (red)')
                           for b in ('local','position','fusion')]
                illustration=np.concatenate(panels,axis=1)
                destination=args.output/(Path(name).stem+'_diagnostic.png')
                cv2.imencode('.png',illustration)[1].tofile(str(destination))
                illustrations.append(str(destination))
        rows.append(row)
        print(f'diagnostic {index}/30 {name}',flush=True)
    summaries={}
    for group in ('fault','normal','damaged','disconnected','misrouted'):
        subset=[r for r in rows if (not r['image'].startswith('normal_') if group=='fault' else r['image'].startswith(group+'_'))]
        summaries[group]={}
        for branch in ('local','position','fusion'):
            keys=list(subset[0][branch])
            summaries[group][branch]={key:float(np.mean([r[branch][key] for r in subset
                                                        if r[branch].get(key) is not None]))
                                      for key in keys if any(r[branch].get(key) is not None for r in subset)}
    result={'posthoc':True,'formal_path_changed':False,'fraction':0.05,
            'border_area_fraction':float(border.mean()),'normal_template_hot_fraction':float(normal_hot['position'].mean()),
            'source_report_sha256':hashlib.sha256((args.maps/'report.json').read_bytes()).hexdigest(),
            'fresh_registration_successes':sum(r['fresh_registration']['alignment_quality']['reliable'] for r in rows),
            'selected_illustrations':selected,'illustrations':illustrations,'summaries':summaries,'cases':rows,
            'warning':'Top fraction is fixed diagnostic, not detector threshold. Normal templates use held-out train calibration only. Source boxes are fragmented/incomplete. Fresh registration metrics and RGB residuals cannot prove original alignment error; residuals also reflect real defects/lighting.'}
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summaries':summaries,'fresh_registration_successes':result['fresh_registration_successes']},indent=2))


if __name__=='__main__':
    main()
