"""Opt-in frozen CNN candidate evidence; never an automatic fault verdict."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def select_with_optional_cnn(baseline, pool, width, height, evidence_factory=None, *, enabled=False):
    """Return old candidates unchanged unless the entire experimental pass succeeds."""
    original=copy.deepcopy(baseline)
    if not enabled:
        return original,{'requested':False,'status':'disabled','baseline_preserved':True}
    if not baseline:
        return original,{'requested':True,'status':'skipped_empty_baseline','baseline_preserved':True}
    try:
        if evidence_factory is None: raise ValueError('CNN evidence provider missing')
        if min(width,height)<=0: raise ValueError('invalid ROI dimensions')
        score,threshold,provenance=evidence_factory()
        score=np.asarray(score)
        if score.shape!=(21,28) or not np.isfinite(score).all(): raise ValueError('invalid CNN score map')
        if not np.isfinite(threshold) or threshold<0: raise ValueError('invalid calibrated threshold')
        from tools.probe_cnn_qualified_support import qualifies
        from tools.probe_local_normal_bank import region_score
        from tools.evaluate_anchored_local import anchored_selection
        def geometry(b):
            v=tuple(b[k] for k in ('left','top','right','bottom'))
            if not all(np.isfinite(v)) or not 0<=v[0]<v[2]<=width or not 0<=v[1]<v[3]<=height:
                raise ValueError('candidate outside frozen ROI')
            return v
        pool_keys={geometry(b) for b in pool}
        if any(geometry(b) not in pool_keys for b in baseline): raise ValueError('baseline outside candidate pool')
        qualified=[b for b in pool if qualifies(b,region_score(b,score,width,height),threshold)]
        # Preserve even an old anchor; all old baseline members must satisfy the frozen gate.
        qualified_keys={geometry(b) for b in qualified}
        if any(geometry(b) not in qualified_keys for b in baseline): raise ValueError('old eligibility no longer matches')
        selected=anchored_selection(baseline,qualified,score,width,height,region_score)
        if len(selected)!=len(baseline) or selected[0]!=baseline[0]: raise ValueError('budget or anchor changed')
        return copy.deepcopy(selected),{'requested':True,'status':'applied','baseline_preserved':selected==baseline,
              'threshold':float(threshold),'qualified_count':len(qualified),'provenance':provenance,
              'evidence_boundary':'possible_difference_manual_review; same-chassis experimental evidence only'}
    except Exception as error:
        return original,{'requested':True,'status':'fallback','baseline_preserved':True,
                         'error_type':type(error).__name__,'reason':str(error).split('\n')[0][:240]}


class FrozenCnnEvidence:
    """Lazy local weights and fingerprinted 80-normal snapshot, fixed reference/ROI."""
    def __init__(self,repo,snapshot,config_dir):
        self.repo=Path(repo); self.snapshot=Path(snapshot); self.config_dir=Path(config_dir)
        self._ready=False

    @staticmethod
    def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def _load(self):
        import importlib.metadata
        import os
        import torch
        source_path=self.snapshot/'original/report.json'
        source=json.loads(source_path.read_text(encoding='utf-8'))
        metric=json.loads((self.snapshot/'metric/report.json').read_text(encoding='utf-8'))
        if metric['source_report_sha256']!=self._sha(source_path): raise ValueError('snapshot report mismatch')
        if source['versions']!={n:importlib.metadata.version(n) for n in source['versions']}: raise ValueError('snapshot runtime versions changed')
        if source['source_sha256']!=self._sha(self.repo/'prototype/tiled_dino_review.py'): raise ValueError('candidate rules changed')
        if source['extractor_sha256']!=self._sha(self.repo/'tools/prepare_cnn_heat_evidence.py'): raise ValueError('CNN extractor changed')
        manifest={(m['split'],m['image']):m for m in source['feature_manifest']}
        if len(source['fit_names'])!=80 or len(set(source['fit_names']))!=80: raise ValueError('requires frozen 80-normal bank')
        canonical=manifest['train01',source['fit_names'][0]]['identity']['canonical']
        alignment_sha=manifest['train01',source['fit_names'][0]]['identity']['alignment']
        if alignment_sha!=self._sha(self.repo/'prototype/assembly_auto_review_robust_v3.py'): raise ValueError('registration code changed')
        frozen_path=self.repo/'output/mendeley_cnn_qualified_support_20260929/report.json'
        frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
        if frozen['experiment_sha256']!=self._sha(self.repo/'tools/probe_cnn_qualified_support.py'): raise ValueError('qualification rules changed')
        if frozen['evidence_report_sha256']!=self._sha(source_path): raise ValueError('frozen source changed')
        calibration_path=self.snapshot/'original/calibration_maps.npz'
        if frozen['calibration_maps_sha256']!=self._sha(calibration_path): raise ValueError('calibration changed')
        with np.load(calibration_path,allow_pickle=False) as data:
            threshold=float(np.percentile(data['fusion'].max(axis=(1,2)),95))
        if threshold!=frozen['threshold']: raise ValueError('threshold mismatch')
        bank=[]
        for name in source['fit_names']:
            entry=manifest['train01',name]; path=self.snapshot/'features'/entry['cache']
            if self._sha(path)!=entry['cache_sha256']: raise ValueError('normal feature cache changed')
            with np.load(path,allow_pickle=False) as data: bank.append(data['features'])
        bank=np.stack(bank)
        if bank.shape!=(80,21,28,192) or not np.isfinite(bank).all(): raise ValueError('normal bank invalid')
        with np.load(self.snapshot/'original/spatial_model.npz',allow_pickle=False) as data:
            model={k:data[k] for k in ('channels','mean')}
        # Verify derived mean rather than trusting an un-fingerprinted model file.
        from tools.prepare_fine_heat_evidence import mean_model
        rebuilt=mean_model(bank,np.array(source['selected_channels']))
        if not np.array_equal(model['channels'],rebuilt['channels']) or not np.allclose(model['mean'],rebuilt['mean'],rtol=0,atol=1e-12):
            raise ValueError('spatial mean changed')
        weights=self.repo/'models/yolov8s-seg.pt'
        if self._sha(weights)!=source['weights_sha256']: raise ValueError('CNN weights changed')
        self.config_dir.mkdir(parents=True,exist_ok=True)
        os.environ['YOLO_CONFIG_DIR']=str(self.config_dir.resolve())
        os.environ['YOLO_OFFLINE']='True'; os.environ['YOLO_AUTOINSTALL']='False'
        from ultralytics import YOLO
        torch.set_num_threads(4)
        network=YOLO(str(weights)).model.cpu().eval(); prefix=list(network.model[:5])
        if [type(l).__name__ for l in prefix]!=['Conv','Conv','C2f','Conv','C2f'] or any(l.f!=-1 for l in prefix):
            raise ValueError('CNN architecture changed')
        self.source=source; self.metric=metric; self.bank=bank; self.model=model; self.prefix=prefix
        self.threshold=threshold; self.canonical=canonical
        self.provenance={'snapshot_sha256':self._sha(source_path),'weights_sha256':source['weights_sha256'],
                         'qualification_sha256':frozen['experiment_sha256'],'runtime_sha256':self._sha(__file__)}
        self._ready=True

    def score(self,reference,aligned,bounds):
        if not self._ready: self._load()
        if hashlib.sha256(reference.tobytes()).hexdigest()!=self.canonical: raise ValueError('unsupported reference scene')
        if list(bounds)!=self.source['bounds'] or aligned.shape!=reference.shape: raise ValueError('unsupported frozen ROI')
        import cv2
        import torch
        from tools.prepare_cnn_heat_evidence import combine_maps
        from tools.probe_local_normal_bank import local_distance
        from tools.probe_spatial_metric import unweighted_score
        from tools.probe_spatial_normal import fuse
        l,t,r,b=bounds; crop=aligned[t:b,l:r]; scale=392/max(crop.shape[:2])
        h,w=max(1,round(crop.shape[0]*scale)),max(1,round(crop.shape[1]*scale))
        resized=cv2.resize(crop,(w,h),interpolation=cv2.INTER_AREA)
        value=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).float()[None]/255
        maps=[]
        with torch.inference_mode():
            for i,layer in enumerate(self.prefix):
                value=layer(value)
                if i in (2,4): maps.append(value)
            query=combine_maps(maps,(21,28))
        score=fuse(local_distance(query,self.bank,k=3,radius=1),unweighted_score(query,self.model),
                   self.source['calibration']['local_scale'],self.metric['calibration']['identity_scale'])
        return score,self.threshold,self.provenance
