"""Recompute fine validation maps/selections from frozen feature caches, no inference/tuning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path('E:/PythonProject10');sys.path.insert(0,str(ROOT))
from tools.run_fine_cnn_validation import nearby_distance
from tools.merge_audit import setup
from tools.prepare_fine_heat_evidence import mean_model
from tools.probe_spatial_metric import unweighted_score
from tools.probe_spatial_normal import fuse
from tools.probe_local_normal_bank import region_score
from tools.probe_cnn_qualified_support import qualifies
from tools.evaluate_anchored_local import anchored_selection
from tools.audit_cnn_precision_ceiling import reconstruct_groups
from inspection_agent.focus_hint import select_focus_hint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    report=load(args.report);assert report['status']=='complete' and report['split']=='val01'
    _,tiled=setup(ROOT);tiled.LARGE_ROI_MAX_CANDIDATES=6
    evidence=args.report.parent;calibration=load(evidence/'calibration.json')
    coarse=load(ROOT/'output/mendeley_cnn_heat_evidence_20260929/original/report.json')
    for key in ('fit_names','embargo_names','calibration_names'):assert coarse[key]==calibration[key]
    manifest={(m['split'],m['image']):m for m in report['manifest']}
    assert len(manifest)==140 and all(m['coarse_max_abs_error']==0 and m['alignment']['alignment_quality']['reliable'] for m in manifest.values())
    def features(split,name):
        m=manifest[split,name];path=evidence/'features'/m['cache']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==m['cache_sha256']
        with np.load(path,allow_pickle=False) as data:return data['features']
    bank=np.stack([features('train01',n) for n in calibration['fit_names']])
    model=mean_model(bank,np.array(coarse['selected_channels']))
    with np.load(evidence/'mean_model.npz',allow_pickle=False) as data:
        assert np.array_equal(data['channels'],model['channels'])
        np.testing.assert_allclose(data['mean'],model['mean'],rtol=0,atol=0)
    normalized=bank/np.maximum(np.linalg.norm(bank,axis=-1,keepdims=True),1e-8);del bank
    with np.load(evidence/'calibration_maps.npz',allow_pickle=False) as data:
        assert float(np.percentile(data['fusion'].max(axis=(1,2)),95))==report['threshold']==calibration['threshold']
    expected={c['image']:c for c in report['cases']};key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
    bounds=coarse['bounds'];x,y,_,_=bounds;verified=[]
    for directory in ('mendeley_merge_audit_20260928','mendeley_normal_evidence_20260928'):
        for path in sorted((ROOT/'output'/directory).glob('*.json')):
            if path.name=='comparison.json':continue
            r=load(path);assert r['split']=='val01' and r['source_sha256']==report['source_sha256']
            t=r['trace'];query=features('val01',r['image'])
            score=fuse(nearby_distance(query,normalized),unweighted_score(query,model),calibration['local_scale'],calibration['position_scale'])
            with np.load(evidence/(path.stem+'_maps.npz'),allow_pickle=False) as data:
                np.testing.assert_allclose(score,data['fusion'],rtol=0,atol=0)
            pool=tiled._merge_candidates(copy.deepcopy(t['raw']));tiled._annotate_roi_edges(pool,t['width'],t['height'])
            pool,suppressed=tiled._publish_candidates(pool,[]);assert not suppressed
            qualified=[b for b in pool if qualifies(b,region_score(b,score,t['width'],t['height']),report['threshold'])]
            selected=anchored_selection(t['local_candidates'],qualified,score,t['width'],t['height'],region_score)
            translate=lambda b:{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y}
            assert [translate(b) for b in selected]==expected[r['image']]['fine_selected']
            groups=reconstruct_groups(t['raw'],tiled);members={key(tiled._merge_candidates(g)[0]):g for g in groups};hints=[]
            for parent in expected[r['image']]['parents']:
                local={**parent,'left':parent['left']-x,'right':parent['right']-x,'top':parent['top']-y,'bottom':parent['bottom']-y}
                hint=select_focus_hint(local,members[key(local)],score,t['width'],t['height'],report['threshold'],expected_grid=(42,56))
                if hint:hints.append(translate(hint['box']))
            assert hints==expected[r['image']]['hints']
            verified.append(r['image']);print(f'replay {len(verified)}/30 {r["image"]} exact',flush=True)
    assert len(verified)==30
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump({'verified':verified,'maps_and_selected_and_hints_exact':True,'coarse_control_exact_140':True,
                   'train_split_preserved':True,'all_registrations_reliable':True,
                   'warning':'Implementation replay only, no new validation tuning.'},file,indent=2)


if __name__=='__main__':main()
