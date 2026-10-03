"""Frozen train-only paired geometry examples; preserve original feature pins."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,OUT,DATA,load,save,sha,read_image
from paired_port_semantics import expected_in_source,valid_boxes,paired_features
from paired_semantic_geometry_examples import perturbations,training_label
from port_semantic_verifier import embeddings
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
sys.path.insert(0,str(REPO/'prototype'))
NEW=ROOT/'artifacts/paired_semantic_localization_20261003'


def main():
    import cv2
    import numpy as np
    import torch
    import dino_feature_diff as dino
    destination=NEW/'features_train'
    if destination.exists():raise FileExistsError('Preserve enriched feature cache')
    base=OUT/'features_train';prepared=load(base/'report.json');assert prepared['status']=='complete' and prepared['gt_targets']==344
    assert {p:sha(Path(p)) for p in prepared['pins']}==prepared['pins']
    assert {p:sha(Path(p)) for p in prepared['cache_pins']}==prepared['cache_pins']
    assert sha(base/'features.pt')==prepared['aggregate_feature_sha256'] and sha(base/'samples.json')==prepared['samples_sha256']
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==prepared['runtime_fingerprint']
    data=torch.load(base/'features.pt',map_location='cpu',weights_only=True);metadata=load(base/'samples.json')
    groups=prepared['source_groups'];assert len(groups)==192
    pins=dict(prepared['pins'])
    for p in (Path(__file__),Path(__file__).with_name('paired_semantic_geometry_examples.py'),base/'features.pt',base/'report.json',base/'samples.json',
              ROOT/'artifacts/paired_semantic_localization_preregistration_20261003/PLAN.md'):
        pins[str(p)]=sha(p)
    for source in prepared['sources']:
        path=base/(Path(source['image']).stem+'_source.json');pins[str(path)]=sha(path)
    destination.mkdir(parents=True)
    save(destination/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,train_sources=groups,
        synthetic_shifts=[[-.75,0],[.75,0],[0,-.75],[0,.75]],synthetic_scales=[.5,2.],all_GT_consistent_training_labels=True,
        no_heldout_training=True,no_GT_or_geometry_metadata_features=True,original1107_samples_preserved=True,
        frozen_encoder=True,no_deployment=True,field_accuracy=False))
    torch.set_num_threads(2);cv2.setNumThreads(2);encoder=dino._model();encoder.requires_grad_(False).eval();torch.set_num_threads(2)
    reference=read_image(DATA/'images/train01/normal_073.JPG');started=time.monotonic()
    features=[data['features']];all_samples=list(metadata);records=[];cache_pins={}
    for index,name in enumerate(groups):
        save(destination/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),
            completed=index,total=len(groups),image=name,samples=len(all_samples)))
        source=load(base/(Path(name).stem+'_source.json'));targets=[r for r in metadata if r['image']==name and r['kind']=='gt_port']
        if not targets:
            records.append(dict(image=name,status='no_GT_training_synthetic_short_circuit',synthetic=0));continue
        image=read_image(DATA/'images/train01'/name)
        assert source['source_sha256']==sha(DATA/'images/train01'/name)
        assert source['alignment']['alignment_quality']['reliable']
        expected,valid=expected_in_source(reference,source['alignment']['source_to_reference_homography'],image.shape[:2])
        truth=[dict(class_id=r['label']-1,box=r['box']) for r in targets]
        rows=[]
        for target in targets:
            for box in perturbations(target['box']):
                label,iou=training_label(box,truth)
                rows.append(dict(image=name,fold=target['fold'],label=label,kind='synthetic_geometry',box=box,
                    training_GT_maximum_iou=iou,source_sha256=source['source_sha256']))
        indices=valid_boxes([r['box'] for r in rows],valid);selected=[rows[i] for i in indices];boxes=[r['box'] for r in selected]
        vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes));assert torch.isfinite(vectors).all()
        path=destination/(Path(name).stem+'_features.pt');torch.save(dict(features=vectors,samples=selected),path);cache_pins[str(path)]=sha(path)
        features.append(vectors);all_samples.extend(selected)
        records.append(dict(image=name,status='features_ready',synthetic=len(selected),invalid_context_abstentions=len(rows)-len(selected),
                            negative=sum(r['label']==0 for r in selected),positive=sum(r['label']>0 for r in selected)))
        save(destination/(Path(name).stem+'_source.json'),records[-1])
    combined=torch.cat(features);labels=torch.tensor([r['label'] for r in all_samples]);folds=torch.tensor([r['fold'] for r in all_samples])
    assert combined.shape==(len(all_samples),6144)
    assert {p:sha(Path(p)) for p in pins}==pins and {p:sha(Path(p)) for p in cache_pins}==cache_pins
    assert resolution_runtime_fingerprint(REPO)==frozen and not any(p.requires_grad for p in encoder.parameters())
    torch.save(dict(features=combined,labels=labels,folds=folds),destination/'features.pt');save(destination/'samples.json',all_samples)
    report=dict(status='complete',samples=len(all_samples),original_samples=len(metadata),gt_targets=344,gt_valid=344,
        counts=torch.bincount(labels,minlength=3).tolist(),source_groups=groups,sources=records,pins=pins,cache_pins=cache_pins,
        aggregate_feature_sha256=sha(destination/'features.pt'),samples_sha256=sha(destination/'samples.json'),
        runtime_fingerprint=frozen,no_validation_training=True,frozen_encoder_unchanged=True,encoder_sha256=prepared['encoder_sha256'],
        seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False)
    save(destination/'report.json',report);save(destination/'progress.json',dict(status='complete',samples=len(all_samples),seconds=report['seconds']))
    print(str({k:v for k,v in report.items() if k not in ('sources','pins','cache_pins','runtime_fingerprint','source_groups')}))


if __name__=='__main__':main()
