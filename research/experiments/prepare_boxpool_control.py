"""Derive explicit controlled subsets from SHA-verified frozen descriptors."""
import copy,sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_boxpool_features import OUT as SOURCE
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint


def prepare(out,*,original_only=False):
    destination=out/'features_train'
    if destination.exists():raise FileExistsError('Preserve controlled data')
    import torch
    torch.set_num_threads(1)
    prepared=load(SOURCE/'features_train/report.json');assert prepared['status']=='complete' and prepared['samples']==4166
    assert paired_runtime_fingerprint(REPO)==prepared['accepted_paired_runtime']
    assert {p:sha(Path(p)) for p in prepared['cache_pins']}==prepared['cache_pins']
    assert sha(SOURCE/'features_train/features.pt')==prepared['aggregate_feature_sha256']
    assert sha(SOURCE/'features_train/samples.json')==prepared['samples_sha256']
    data=torch.load(SOURCE/'features_train/features.pt',map_location='cpu',weights_only=True);rows=load(SOURCE/'features_train/samples.json')
    indices=[i for i,r in enumerate(rows) if not original_only or r['kind']!='synthetic_aspect'];metadata=[rows[i] for i in indices]
    assert len(metadata)==(3134 if original_only else 4166) and sum(r['kind']=='gt_port' for r in metadata)==344
    values=data['features'][indices];labels=data['labels'][indices];folds=data['folds'][indices]
    destination.mkdir(parents=True);pins=dict(prepared['pins'])
    for path in (Path(__file__),SOURCE/'features_train/features.pt',SOURCE/'features_train/samples.json',SOURCE/'features_train/report.json',
        ROOT/'artifacts/paired_boxpool_controls_preregistration_20261003/PLAN.md'):
        pins[str(path)]=sha(path)
    sources=[];cache_pins={}
    for old in prepared['sources']:
        name=old['image'];local=[i for i,r in enumerate(metadata) if r['image']==name];samples=[metadata[i] for i in local]
        feature_path=destination/(Path(name).stem+'_features.pt');torch.save(dict(features=values[local],samples=samples),feature_path);cache_pins[str(feature_path)]=sha(feature_path)
        row=copy.deepcopy(old);row.update(features=str(feature_path),features_sha256=cache_pins[str(feature_path)],samples=len(samples),
            controlled_feature_subset=True,synthetic_aspect_removed=bool(original_only))
        sources.append(row);save(destination/(Path(name).stem+'_source.json'),row)
    torch.save(dict(features=values,labels=labels,folds=folds),destination/'features.pt');save(destination/'samples.json',metadata)
    result=copy.deepcopy(prepared);result.update(samples=len(metadata),sources=sources,pins=pins,cache_pins=cache_pins,
        counts=torch.bincount(labels,minlength=3).tolist(),aggregate_feature_sha256=sha(destination/'features.pt'),
        samples_sha256=sha(destination/'samples.json'),control_derives_same_frozen_features=True,
        no_new_encoder_inference=True,original_rows_only=bool(original_only))
    save(destination/'report.json',result)
    assert {p:sha(Path(p)) for p in pins}==pins

