"""Frozen local CNN prototype using the existing generic YOLOv8 backbone."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def combine_maps(maps,grid):
    if len(maps)!=2 or min(grid)<=0:
        raise ValueError('invalid feature pyramid')
    pooled=[]
    for value in maps:
        if value.ndim!=4 or value.shape[0]!=1 or not torch.isfinite(value).all():
            raise ValueError('invalid feature map')
        value=F.avg_pool2d(value,kernel_size=3,stride=1,padding=1)
        value=F.adaptive_avg_pool2d(value,grid)
        value=F.normalize(value,dim=1,eps=1e-8)
        pooled.append(value)
    result=torch.cat(pooled,dim=1)[0].permute(1,2,0).contiguous()
    return result.cpu().numpy().astype(np.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    parser.add_argument('--coarse-maps',type=Path,required=True)
    parser.add_argument('--fault-cache',type=Path,required=True)
    parser.add_argument('--normal-cache',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new CNN evidence directory')
    if not args.weights.is_file():
        raise FileNotFoundError('explicit local weights required; no download')
    args.output.mkdir(parents=True)
    config=args.output/'ultralytics_config'; config.mkdir()
    os.environ['YOLO_CONFIG_DIR']=str(config.resolve())
    os.environ['YOLO_OFFLINE']='True'; os.environ['YOLO_AUTOINSTALL']='False'
    from ultralytics import YOLO
    torch.set_num_threads(4); torch.manual_seed(20260929); cv2.setRNGSeed(0)
    yolo=YOLO(str(args.weights.resolve()))
    network=yolo.model.cpu().eval()
    prefix=list(network.model[:5])
    assert all(layer.f==-1 for layer in prefix)
    assert [type(layer).__name__ for layer in prefix]==['Conv','Conv','C2f','Conv','C2f']
    sys.path.insert(0,str(args.repo))
    from tools.merge_audit import setup,save
    from tools.prepare_fine_heat_evidence import mean_model
    from tools.probe_spatial_metric import unweighted_score
    from tools.probe_local_normal_bank import local_distance
    from tools.probe_spatial_normal import normal_scale,fuse
    setup(args.repo)
    from evaluate_mendeley_balanced import read_image
    import assembly_auto_review_robust_v3 as perspective
    source=json.loads((args.coarse_maps/'report.json').read_text(encoding='utf-8'))
    bounds=source['bounds']; left,top,right,bottom=bounds; grid=(21,28)
    dataset=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    reference=read_image(dataset/'images/train01/normal_073.JPG')
    weight_hash=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    extractor_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    alignment_hash=hashlib.sha256((args.repo/'prototype/assembly_auto_review_robust_v3.py').read_bytes()).hexdigest()
    canonical_hash=hashlib.sha256(reference.tobytes()).hexdigest()
    original=args.output/'original'; metric=args.output/'metric'; feature_dir=args.output/'features'
    original.mkdir(); metric.mkdir(); feature_dir.mkdir()
    manifest=[]; alignments=[]
    versions={p:importlib.metadata.version(p) for p in ('torch','torchvision','ultralytics')}

    def features(split,name):
        path=dataset/'images'/split/name
        cv2.setRNGSeed(0)
        aligned,alignment=perspective.automatic_homography(reference,read_image(path))
        if aligned is None:
            raise RuntimeError(f'CNN registration failed: {split}/{name}')
        crop=aligned[top:bottom,left:right]
        scale=392/max(crop.shape[:2])
        height,width=max(1,round(crop.shape[0]*scale)),max(1,round(crop.shape[1]*scale))
        resized=cv2.resize(crop,(width,height),interpolation=cv2.INTER_AREA)
        tensor=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).float()[None]/255
        maps=[]
        with torch.inference_mode():
            for index,layer in enumerate(prefix):
                tensor=layer(tensor)
                if index in (2,4):
                    maps.append(tensor)
            value=combine_maps(maps,grid)
        assert value.shape==(21,28,192)
        identity={'path':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'weights':weight_hash,'extractor':extractor_hash,'alignment':alignment_hash,
                  'canonical':canonical_hash,'bounds':bounds,'grid':grid,'input':(height,width),'versions':versions}
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        cached=feature_dir/(fingerprint+'.npz')
        np.savez_compressed(cached,features=value)
        manifest.append({'split':split,'image':name,'cache':cached.name,
                         'cache_sha256':hashlib.sha256(cached.read_bytes()).hexdigest(),
                         'identity':identity})
        alignments.append({'split':split,'image':name,'alignment':alignment})
        return value

    fit=[]
    for index,name in enumerate(source['fit_names'],1):
        fit.append(features('train01',name))
        print(f'CNN fitting {index}/80 {name}',flush=True)
    bank=np.stack(fit); del fit
    channels=np.sort(np.random.default_rng(20260928).choice(192,32,replace=False))
    model=mean_model(bank,channels)
    local_cal=[]; identity_cal=[]
    for index,name in enumerate(source['calibration_names'],1):
        query=features('train01',name)
        local_cal.append(local_distance(query,bank,k=3,radius=1))
        identity_cal.append(unweighted_score(query,model))
        print(f'CNN calibration {index}/30 {name}',flush=True)
    local_scale,local_info=normal_scale(local_cal)
    identity_scale,identity_info=normal_scale(identity_cal)
    fused_cal=[fuse(l,i,local_scale,identity_scale) for l,i in zip(local_cal,identity_cal)]
    precision=np.broadcast_to(np.eye(32),grid+(32,32))
    np.savez_compressed(original/'spatial_model.npz',**model,precision=precision)
    np.savez_compressed(original/'calibration_maps.npz',local=np.stack(local_cal),identity=np.stack(identity_cal),fusion=np.stack(fused_cal))
    records=[]
    for directory in (args.fault_cache,args.normal_cache):
        records += [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('*.json'))
                    if p.name!='comparison.json']
    source_hash=hashlib.sha256((args.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert len(records)==30 and source_hash==source['source_sha256']
    assert all(r['split']=='val01' and r['bounds']==bounds and r['source_sha256']==source_hash for r in records)
    for index,r in enumerate(records,1):
        query=features('val01',r['image'])
        local=local_distance(query,bank,k=3,radius=1)
        identity=unweighted_score(query,model)
        fusion=fuse(local,identity,local_scale,identity_scale)
        np.savez_compressed(metric/(Path(r['image']).stem+'_maps.npz'),identity=identity,fusion=fusion)
        print(f'CNN validation {index}/30 {r["image"]}',flush=True)
    save(original/'report.json',{'protocol':'cnn_prefix_layers2_4_normal80_v1','grid':[21,28,192],
         'bounds':bounds,'source_sha256':source_hash,'fit_names':source['fit_names'],
         'embargo_names':source['embargo_names'],'calibration_names':source['calibration_names'],
         'feature_manifest':manifest,'alignments':alignments,'calibration':{'local_scale':local_scale,'local':local_info},
         'model_mode':'frozen generic YOLO prefix; unweighted spatial normal mean, not PatchCore',
         'checkpoint_data':yolo.model.args.get('data'),'checkpoint_version':yolo.ckpt.get('version'),
         'weights_sha256':weight_hash,'extractor_sha256':extractor_hash,'versions':versions,
         'input_max_edge':392,'layers':[2,4],'local_pool':3,'branch_normalization':'L2 per layer after adaptive pooling',
         'seed':20260928,'selected_channels':channels.tolist(),'formal_path_changed':False,
         'warning':'Fresh registration differs from legacy cached transform; CNN architecture and preprocessing differ from DINO. Controlled prototype, not a pure identical-feature swap or field accuracy.'})
    save(metric/'report.json',{'source_report_sha256':hashlib.sha256((original/'report.json').read_bytes()).hexdigest(),
         'calibration':{'identity_scale':identity_scale,'identity':identity_info},'formal_path_changed':False})
    print(json.dumps({'grid':[21,28,192],'normal_peak95':float(np.percentile([m.max() for m in fused_cal],95)),
                      'weights_sha256':weight_hash,'manifest_count':len(manifest)},indent=2))


if __name__=='__main__':
    main()
