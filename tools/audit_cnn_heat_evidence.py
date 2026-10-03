"""Post-hoc CNN signal and registration-coordinate audit; never retunes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--fault-cache',type=Path,required=True)
    parser.add_argument('--dino-maps',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('use a new audit directory')
    sys.path.insert(0,str(args.repo)); sys.path.insert(0,str(args.repo/'prototype'))
    from tools.audit_spatial_evidence import target_grid,rank_auc
    from tools.diagnose_spatial_hotspots import panel,top_mask
    from tools.evaluate_mendeley_holdout_current import transform_boxes
    from tools.merge_audit import save
    from evaluate_mendeley_balanced import read_image,load_target_boxes
    import assembly_auto_review_robust_v3 as perspective
    source=json.loads((args.evidence/'original/report.json').read_text(encoding='utf-8'))
    extraction_alignments={e['image']:e['alignment'] for e in source['alignments'] if e['split']=='val01'}
    candidate_report=json.loads((args.candidates/'report.json').read_text(encoding='utf-8'))
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.fault_cache.glob('*.json'))
             if p.name!='comparison.json']
    assert len(records)==15 and all(r['split']=='val01' for r in records)
    selected={kind:min(r['image'] for r in records if r['image'].startswith(kind+'_'))
              for kind in ('damaged','disconnected','misrouted')}
    normal_names=[r['image'] for r in candidate_report['results']['baseline']['cases'] if r['image'].startswith('normal_')]
    selected['normal']=min(normal_names)
    dataset=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    reference=read_image(dataset/'images/train01/normal_073.JPG')
    args.output.mkdir(parents=True)
    rows=[]; illustrations=[]
    original_find=cv2.findHomography
    captured=[]

    def capture(*a,**kw):
        result=original_find(*a,**kw)
        captured.append(result[0])
        return result

    try:
        cv2.findHomography=capture
        for index,record in enumerate(records+[{'image':selected['normal'],'targets':[]}],1):
            name=record['image']; inspection=read_image(dataset/'images/val01'/name)
            captured.clear(); cv2.setRNGSeed(0)
            aligned,alignment=perspective.automatic_homography(reference,inspection)
            assert aligned is not None and len(captured)==1 and captured[0] is not None
            targets=transform_boxes(load_target_boxes(dataset/'labels/val01'/(Path(name).stem+'.txt'),
                                                      inspection.shape[1],inspection.shape[0]),captured[0])
            row={'image':name,'cached_targets_exact':targets==record['targets'],
                 'extraction_alignment_report_exact':alignment==extraction_alignments[name],
                 'actual_homography':captured[0].tolist(),'alignment':alignment,
                 'fresh_targets':targets,'cached_targets':record['targets']}
            with np.load(args.evidence/'metric'/(Path(name).stem+'_maps.npz'),allow_pickle=False) as data:
                maps={branch:data[branch] for branch in ('identity','fusion')}
            mask=target_grid(targets,source['bounds'],maps['fusion'].shape)
            row['cnn_grid_overlap_rank']={b:rank_auc(m,mask) for b,m in maps.items()}
            left,top,right,bottom=source['bounds']
            valid=perspective.auto.LAST_WARP_VALID_MASK[top:bottom,left:right]>0
            coverage=cv2.resize(valid.astype(np.float32),(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_AREA)
            row['valid_roi_fraction']=float(valid.mean())
            row['cnn_hot_invalid_warp_fraction']={b:float(np.mean((coverage<0.99)[top_mask(m)]))
                                                 for b,m in maps.items()}
            if targets:
                with np.load(args.dino_maps/(Path(name).stem+'_maps.npz'),allow_pickle=False) as data:
                    row['dino_grid_overlap_rank']={b:rank_auc(data[b],mask) for b in ('identity','fusion')}
            if name in selected.values():
                panels=[panel(aligned,None,targets,source['bounds'],name+'; cyan=source boxes')]
                panels += [panel(aligned,top_mask(maps[b]),targets,source['bounds'],'CNN '+b+' top 5% cells')
                           for b in ('identity','fusion')]
                illustration=np.concatenate(panels,axis=1)
                destination=args.output/(Path(name).stem+'_diagnostic.png')
                cv2.imencode('.png',illustration)[1].tofile(str(destination)); illustrations.append(str(destination))
            rows.append(row)
            print(f'CNN coordinate audit {index}/16 {name} exact={row["cached_targets_exact"]}',flush=True)
    finally:
        cv2.findHomography=original_find
    summaries={}
    for kind in ('all','damaged','disconnected','misrouted'):
        subset=[r for r in rows if r['fresh_targets'] and (kind=='all' or r['image'].startswith(kind+'_'))]
        summaries[kind]={model:{branch:float(np.mean([r[model+'_grid_overlap_rank'][branch] for r in subset]))
                               for branch in ('identity','fusion')} for model in ('cnn','dino')}
    result={'posthoc':True,'parameters_changed':False,'formal_path_changed':False,
            'cached_target_exact_fault_images':sum(r['cached_targets_exact'] for r in rows if r['fresh_targets']),
            'extraction_alignment_reports_exact':sum(r['extraction_alignment_report_exact'] for r in rows),
            'summaries':summaries,'selected_illustrations':selected,'illustrations':illustrations,'cases':rows,
            'evidence_report_sha256':hashlib.sha256((args.evidence/'original/report.json').read_bytes()).hexdigest(),
            'warning':'Fresh deterministic transform coordinate check, not original transform recovery. Rank uses coarse cells intersecting fragmented source boxes, not defect AUROC. Repeated validation and same chassis cannot establish generalization.'}
    save(args.output/'report.json',result)
    print(json.dumps({'coordinate_exact':result['cached_target_exact_fault_images'],'summaries':summaries},indent=2))


if __name__=='__main__': main()
