"""Prepare fine-grid evidence for the identical normal-calibrated peak protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def mean_model(bank,channels):
    bank=np.asarray(bank)
    channels=np.asarray(channels)
    if bank.ndim!=4 or min(bank.shape)<=0 or channels.ndim!=1 or not len(channels):
        raise ValueError('invalid normal feature geometry')
    if not np.isfinite(bank).all() or min(channels)<0 or max(channels)>=bank.shape[-1]:
        raise ValueError('invalid features or channels')
    mean=np.zeros(bank.shape[1:3]+(len(channels),),dtype=np.float64)
    for feature in bank:
        value=feature.astype(np.float64)
        value/=np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-8)
        mean+=value[...,channels]/len(bank)
    return {'channels':channels.copy(),'mean':mean}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    parser.add_argument('--coarse-maps',type=Path,required=True)
    parser.add_argument('--feature-cache',type=Path,required=True)
    parser.add_argument('--fault-cache',type=Path,required=True)
    parser.add_argument('--normal-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new evidence directory')
    sys.path.insert(0,str(args.repo))
    from tools.merge_audit import setup,save
    from tools.probe_local_normal_bank import local_distance
    from tools.probe_spatial_metric import unweighted_score
    from tools.probe_spatial_normal import normal_scale,fuse
    setup(args.repo)
    from evaluate_mendeley_balanced import read_image
    coarse=json.loads((args.coarse_maps/'report.json').read_text(encoding='utf-8'))
    legacy=json.loads((args.feature_cache/'report.json').read_text(encoding='utf-8'))
    assert legacy['feature_max_edge']==784 and legacy['grid']==[41,56,384]
    assert sorted(coarse['fit_names']+coarse['embargo_names']+coarse['calibration_names'])==legacy['bank_names']
    bounds=coarse['bounds']; grid=tuple(legacy['grid'])
    source_hash=hashlib.sha256((args.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert source_hash==coarse['source_sha256']
    dataset=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    canonical=read_image(dataset/'images/train01/normal_073.JPG')
    canonical_hash=hashlib.sha256(canonical.tobytes()).hexdigest()
    manifest=[]

    def features(split,name):
        path=dataset/'images'/split/name; stat=path.stat()
        identity=hashlib.sha256((str(path.resolve())+str(stat.st_size)+str(stat.st_mtime_ns)
                                +str(bounds)+str(grid)+canonical_hash+'local_bank_v1').encode()).hexdigest()
        cached=args.feature_cache/(identity+'.npz')
        if not cached.is_file():
            raise FileNotFoundError(f'fine feature cache missing for {split}/{name}')
        with np.load(cached,allow_pickle=False) as data:
            value=data['features']
        assert value.shape==grid and np.isfinite(value).all()
        manifest.append({'split':split,'image':name,'cache':cached.name,
                         'cache_sha256':hashlib.sha256(cached.read_bytes()).hexdigest()})
        return value

    bank=np.stack([features('train01',name) for name in coarse['fit_names']])
    with np.load(args.coarse_maps/'spatial_model.npz',allow_pickle=False) as data:
        channels=data['channels']
    model=mean_model(bank,channels)
    local_cal=[]; identity_cal=[]
    for index,name in enumerate(coarse['calibration_names'],1):
        query=features('train01',name)
        local_cal.append(local_distance(query,bank,k=3,radius=2))
        identity_cal.append(unweighted_score(query,model))
        print(f'fine calibration {index}/30 {name}',flush=True)
    local_scale,local_info=normal_scale(local_cal)
    identity_scale,identity_info=normal_scale(identity_cal)
    fused_cal=[fuse(l,i,local_scale,identity_scale) for l,i in zip(local_cal,identity_cal)]
    args.output.mkdir(parents=True)
    original=args.output/'original'; metric=args.output/'metric'
    original.mkdir(); metric.mkdir()
    # Identity matrix exists solely for compatibility with the shared reader.
    # It is not a fitted Gaussian precision matrix.
    precision=np.broadcast_to(np.eye(len(channels)),grid[:2]+(len(channels),len(channels)))
    np.savez_compressed(original/'spatial_model.npz',**model,precision=precision)
    np.savez_compressed(original/'calibration_maps.npz',local=np.stack(local_cal),identity=np.stack(identity_cal),fusion=np.stack(fused_cal))
    records=[]
    for directory in (args.fault_cache,args.normal_cache):
        records += [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('*.json'))
                    if p.name!='comparison.json']
    assert len(records)==30 and all(r['split']=='val01' and r['bounds']==bounds and r['source_sha256']==source_hash for r in records)
    for index,record in enumerate(records,1):
        query=features('val01',record['image'])
        local=local_distance(query,bank,k=3,radius=2)
        identity=unweighted_score(query,model)
        fusion=fuse(local,identity,local_scale,identity_scale)
        np.savez_compressed(metric/(Path(record['image']).stem+'_maps.npz'),identity=identity,fusion=fusion)
        print(f'fine validation {index}/30 {record["image"]}',flush=True)
    save(original/'report.json',{'protocol':'same_groups_channels_metric_fine_grid_v1',
         'grid':list(grid),'feature_max_edge':784,'bounds':bounds,'source_sha256':source_hash,
         'fit_names':coarse['fit_names'],'embargo_names':coarse['embargo_names'],'calibration_names':coarse['calibration_names'],
         'feature_manifest':manifest,'calibration':{'local_scale':local_scale,'local':local_info},
         'local_k':3,'local_radius':2,'model_mode':'same selected channels, unweighted distance to spatial normal mean',
         'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'legacy_cache_warning':'Original cache lacks model version in identity; actual cache hashes recorded.',
         'formal_path_changed':False})
    save(metric/'report.json',{'source_report_sha256':hashlib.sha256((original/'report.json').read_bytes()).hexdigest(),
         'calibration':{'identity_scale':identity_scale,'identity':identity_info},'formal_path_changed':False})
    print(json.dumps({'local_scale':local_scale,'identity_scale':identity_scale,'grid':grid,
                      'calibrated_peak95':float(np.percentile([m.max() for m in fused_cal],95))},indent=2))


if __name__=='__main__':
    main()
