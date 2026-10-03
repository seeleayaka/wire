"""Replay all tiled geometry, calibration and hint selection without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def transform_boxes(boxes,matrix):
    out=[]
    for l,t,r,b in boxes:
        c=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
        out.append([int(np.floor(c[:,0].min())),int(np.floor(c[:,1].min())),
                    int(np.ceil(c[:,0].max())),int(np.ceil(c[:,1].max()))])
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    sys.path.insert(0,str(args.repo))
    from inspection_agent.port_tiling import tile_windows,near_artificial_edge,merge_tiled_ports,select_additional_tile_hint,box_iou
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    report=load(args.report);assert report['status']=='complete' and report['split']=='val01'
    assert (report['tile_size'],report['stride'],report['edge_margin'],report['cross_tile_nms_iou'],report['maximum_new_hints_per_image'])==(1280,960,16,.5,1)
    previous=load(args.repo/'output/port_state_hints_validation_20260929/report.json')
    old_cases={c['image']:c for c in previous['cases']}
    assert len(report['source_manifest'])==60 and len(report['calibration'])==30 and len(report['cases'])==30
    assert {c['image'] for c in report['cases']}==set(old_cases)
    for path,digest in report['fingerprints'].items():assert sha(path)==digest,path
    data=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
    reference=cv2.imdecode(np.fromfile(str(data/'train01/normal_073.JPG'),dtype=np.uint8),cv2.IMREAD_COLOR)
    cached={}
    for item in report['source_manifest']:
        assert item['split'] in ('train01','val01')
        cache_path=args.report.parent/'cache'/item['cache']
        assert sha(cache_path)==item['cache_sha256'] and sha(data/item['split']/item['image'])==item['source_sha256']
        cache=load(cache_path);h,w=cache['shape']
        assert cache['windows']==[list(v) for v in tile_windows(w,h)]
        assert cache['raw_prediction_count']==len(cache['edge_kept_predictions'])+cache['edge_rejected_count']
        for prediction in cache['edge_kept_predictions']:
            window=cache['windows'][prediction['source_tile']]
            l,t,r,b=prediction['box_xyxy'];x,y,rr,bb=window
            assert 0<=l<r<=w and 0<=t<b<=h
            assert not near_artificial_edge([l-x,t-y,r-x,b-y],window,w,h)
        merged=merge_tiled_ports(cache['edge_kept_predictions'])
        assert merged==cache['merged_predictions']
        assert merged==merge_tiled_ports(list(reversed(cache['edge_kept_predictions'])))
        cached[item['split'],item['image']]=cache
    for sample in report['calibration']:
        assert sample['image'].startswith('normal_')
        assert sample['maximum_confidence']==max((r['confidence'] for r in cached['train01',sample['image']]['merged_predictions']),default=0.)
    threshold=max(.25,float(np.quantile([c['maximum_confidence'] for c in report['calibration']],.95)))
    assert report['threshold']==threshold
    for case in report['cases']:
        old=old_cases[case['image']]
        assert case['parents']==old['parents'] and case['existing_hints']==old['hints']
        assert case['targets']==old['targets'] and case['source_classes']==old['source_classes']
        assert case['actual_homography']==old['actual_homography']
        cache=cached['val01',case['image']];h,w=cache['shape'];matrix=np.asarray(case['actual_homography'])
        valid=cv2.warpPerspective(np.full((h,w),255,np.uint8),matrix,(reference.shape[1],reference.shape[0]),
            flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        valid=cv2.erode(valid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
        transformed=[]
        for raw,bounds in zip(cache['merged_predictions'],transform_boxes([p['box_xyxy'] for p in cache['merged_predictions']],matrix)):
            l,t,r,b=bounds;crop=valid[max(0,t):min(valid.shape[0],b),max(0,l):min(valid.shape[1],r)]
            coverage=float(crop.sum()/max(1,(r-l)*(b-t))) if crop.size else 0.
            transformed.append({**dict(zip(('left','top','right','bottom'),bounds)),
                'class_id':raw['class_id'],'confidence':raw['confidence'],'valid_warp_fraction':coverage,'support_tiles':raw['support_tiles']})
        assert transformed==case['aligned_predictions']
        selection=select_additional_tile_hint(case['parents'],case['existing_hints'],transformed,threshold)
        assert selection=={k:case[k] for k in ('parents','existing_hints','tile_hints','selection_audit')}
        reversed_selection=select_additional_tile_hint(case['parents'],case['existing_hints'],list(reversed(transformed)),threshold)
        assert reversed_selection['tile_hints']==case['tile_hints']
    def summary(added):
        faults=[c for c in report['cases'] if c['targets']];normals=[c for c in report['cases'] if not c['targets']]
        boxes=lambda c:c['parents']+[h['box'] for h in c['existing_hints']]+([h['box'] for h in c['tile_hints']] if added else [])
        coords=lambda b:[b[k] for k in ('left','top','right','bottom')]
        per=lambda c:[max((box_iou(coords(b),t) for b in boxes(c)),default=0.) for t in c['targets']]
        values=[v for c in faults for v in per(c)]
        return {'fault_images_with_overlap':sum(any(v>0 for v in per(c)) for c in faults),'source_fragments':len(values),
            'overlap_fragments':sum(v>0 for v in values),'iou_ge_01':sum(v>=.1 for v in values),'iou_ge_05':sum(v>=.5 for v in values),
            'mean_best_iou':float(np.mean(values)),'fault_regions':sum(len(boxes(c)) for c in faults),
            'normal_regions':sum(len(boxes(c)) for c in normals),'normal_images_with_added_hints':sum(bool(c['tile_hints']) for c in normals) if added else 0}
    baseline=summary(False); tiled=summary(True)
    # Preserve the frozen run verbatim. Only its baseline diagnostic counter
    # inadvertently reused the addition count; region/IoU metrics are unaffected.
    raw_baseline=dict(report['baseline'])
    assert raw_baseline.pop('normal_images_with_added_hints')==tiled['normal_images_with_added_hints']
    assert raw_baseline=={k:v for k,v in baseline.items() if k!='normal_images_with_added_hints'}
    assert tiled==report['with_tiled_hints']
    assert report['model_tile_passes']==sum(len(c['windows']) for c in cached.values())==720
    gates={key:sum(c['selection_audit'][key] for c in report['cases']) for key in
           ('prediction_count','invalid','below_threshold','invalid_warp','no_containing_smaller_parent','eligible','hint_count')}
    above=[{'image':c['image'],'kind':c['kind'],'prediction':p} for c in report['cases']
           for p in c['aligned_predictions'] if p['confidence']>threshold]
    result={'cache_and_source_hashes_verified':60,'cross_tile_nms_exact_replay':60,
        'hint_and_warp_exact_replay':30,'reversed_order_stable':True,'parents_and_existing_hints_unchanged':True,
        'threshold_rebuilt_exact':True,'region_and_iou_summaries_exact':True,'model_tile_passes':report['model_tile_passes'],
        'corrected_baseline':baseline,'verified_tiled_summary':tiled,
        'reporting_correction':'Raw baseline normal_images_with_added_hints reused addition count; baseline is zero. No geometry, IoU or inference changed.',
        'no_model_inference':True,'validation_gate_counts':gates,'above_threshold_predictions':above}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
