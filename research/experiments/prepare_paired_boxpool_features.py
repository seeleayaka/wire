"""Fresh frozen footprint descriptors over exact existing TRAIN samples."""
import argparse,copy,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,OUT as ORIGINAL,load,save,sha,read_image
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from paired_port_semantics import expected_in_source,valid_boxes,paired_features
from paired_boxpool_features import embeddings
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
OUT=ROOT/'artifacts/paired_boxpool_20261003'


def main(mode):
    destination=OUT/('smoke' if mode=='smoke' else 'features_train')
    if destination.exists():raise FileExistsError('Preserve footprint descriptors')
    sample_path=ROOT/'artifacts/paired_shape_train_probe_20261003/samples.json'
    rows=load(sample_path);prepared=load(GEOMETRY/'features_train/report.json')
    assert len(rows)==4166 and sum(r['kind']=='gt_port' for r in rows)==344
    assert prepared['no_validation_training'] and prepared['frozen_encoder_unchanged']
    groups=prepared['source_groups'];assert len(groups)==192
    if mode=='smoke':groups=[n for n in groups if any(r['image']==n and r['kind']=='gt_port' for r in rows)][:2]
    else:assert load(OUT/'smoke/report.json')['real_footprint_inference_passed']
    old_runtime=resolution_runtime_fingerprint(REPO);assert old_runtime==prepared['runtime_fingerprint']
    frozen=paired_runtime_fingerprint(REPO);pins=dict(prepared['pins'])
    for path in (sample_path,GEOMETRY/'features_train/report.json',Path(__file__),Path(__file__).with_name('paired_boxpool_features.py'),
        Path(__file__).with_name('paired_shape_features.py'),Path(__file__).with_name('paired_port_semantics.py'),
        ROOT/'artifacts/paired_boxpool_preregistration_20261003/PLAN.md'):
        pins[str(path)]=sha(path)
    for name in groups:
        path=ORIGINAL/'features_train'/(Path(name).stem+'_source.json');pins[str(path)]=sha(path)
    assert {p:sha(Path(p)) for p in pins}==pins
    destination.mkdir(parents=True)
    save(destination/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,original_runtime=old_runtime,mode=mode,
        train_sources=groups,no_validation_training=True,footprint_area_weighted16grid=True,
        contexts=[1.5,3.],feature_dimensions=6144,no_metadata_channels=True,
        reference_self_pairs_expected_expected=True,exact_signature_forward_deduplication=True,
        encoder_sha256=prepared['encoder_sha256'],no_deployment=True,field_accuracy=False))
    import cv2,numpy as np,torch
    import dino_feature_diff as dino
    cv2.setNumThreads(2);torch.set_num_threads(2);model=dino._model();model.requires_grad_(False).eval();torch.set_num_threads(2)
    encoder_path=REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth';assert sha(encoder_path)==prepared['encoder_sha256']
    reference=read_image(DATA/'images/train01/normal_073.JPG');started=time.monotonic()
    values=[];metadata=[];records=[];cache_pins={};unique=0;requested=0;real_differences=0
    try:
        for index,name in enumerate(groups):
            local=[copy.deepcopy(r) for r in rows if r['image']==name]
            source_path=ORIGINAL/'features_train'/(Path(name).stem+'_source.json');original=load(source_path)
            save(destination/'progress.json',dict(status='running',pid=os.getpid(),completed=index,total=len(groups),image=name,
                samples=len(metadata),phase='fresh_frozen_footprint_embeddings',seconds=round(time.monotonic()-started,2)))
            record=copy.deepcopy(original)
            if local:
                path=DATA/'images/train01'/name;assert sha(path)==original['source_sha256'];image=read_image(path)
                assert original['alignment']['alignment_quality']['reliable']
                expected,valid=expected_in_source(reference,original['alignment']['source_to_reference_homography'],image.shape[:2])
                boxes=[r['box'] for r in local];assert valid_boxes(boxes,valid)==list(range(len(boxes)))
                observed_audit={};expected_audit={}
                observed=embeddings(model,image,boxes,audit=observed_audit);expected_vectors=embeddings(model,expected,boxes,audit=expected_audit)
                self_indices=[i for i,r in enumerate(local) if r['kind']=='reference_self']
                if self_indices:observed[self_indices]=expected_vectors[self_indices]
                vectors=paired_features(observed,expected_vectors)
                assert torch.isfinite(vectors).all()
                for a,row in enumerate(local):
                    if row['kind']!='synthetic_aspect':continue
                    gt=next((i for i,r in enumerate(local) if r['kind']=='gt_port' and r['box'][0]<=row['box'][0] and r['box'][2]>=row['box'][2]
                        and np.allclose([(r['box'][0]+r['box'][2])/2,(r['box'][1]+r['box'][3])/2],[(row['box'][0]+row['box'][2])/2,(row['box'][1]+row['box'][3])/2],atol=1e-7)),None)
                    if gt is not None and not torch.equal(vectors[gt],vectors[a]):real_differences+=1
                unique+=observed_audit['unique_crops']+expected_audit['unique_crops']
                requested+=observed_audit['requested_crops']+expected_audit['requested_crops']
                record.update(footprint_observed=observed_audit,footprint_expected=expected_audit,status='features_ready')
            else:vectors=torch.empty((0,6144));record.update(status='no_valid_sample_exact_short_circuit')
            feature_path=destination/(Path(name).stem+'_features.pt');torch.save(dict(features=vectors,samples=local),feature_path);cache_pins[str(feature_path)]=sha(feature_path)
            record.update(samples=len(local),features=str(feature_path),features_sha256=cache_pins[str(feature_path)])
            save(destination/(Path(name).stem+'_source.json'),record);records.append(record);values.append(vectors);metadata.extend(local)
        combined=torch.cat(values);labels=torch.tensor([r['label'] for r in metadata]);folds=torch.tensor([r['fold'] for r in metadata])
        assert combined.shape==(len(metadata),6144) and torch.isfinite(combined).all()
        assert real_differences>0 and unique<requested
        assert not any(p.requires_grad for p in model.parameters()) and sha(encoder_path)==prepared['encoder_sha256']
        assert {p:sha(Path(p)) for p in pins}==pins and {p:sha(Path(p)) for p in cache_pins}==cache_pins
        assert paired_runtime_fingerprint(REPO)==frozen and resolution_runtime_fingerprint(REPO)==old_runtime
        torch.save(dict(features=combined,labels=labels,folds=folds),destination/'features.pt');save(destination/'samples.json',metadata)
        result=dict(status='complete',mode=mode,samples=len(metadata),gt_targets=sum(r['kind']=='gt_port' for r in metadata),
            gt_valid=sum(r['kind']=='gt_port' for r in metadata),counts=torch.bincount(labels,minlength=3).tolist(),source_groups=groups,sources=records,
            pins=pins,cache_pins=cache_pins,aggregate_feature_sha256=sha(destination/'features.pt'),samples_sha256=sha(destination/'samples.json'),
            runtime_fingerprint=old_runtime,accepted_paired_runtime=frozen,encoder_sha256=prepared['encoder_sha256'],
            no_validation_training=True,frozen_encoder_unchanged=True,real_footprint_inference_passed=True,
            real_short_axis_descriptor_differences=real_differences,unique_encoder_crops=unique,requested_encoder_crops=requested,
            seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False)
        save(destination/'report.json',result);save(destination/'progress.json',dict(status='complete',samples=len(metadata),seconds=result['seconds']))
        print(str({k:v for k,v in result.items() if k not in ('sources','pins','cache_pins','source_groups','runtime_fingerprint','accepted_paired_runtime')}),flush=True)
    except BaseException as error:
        save(destination/'progress.json',dict(status='failed',pid=os.getpid(),error=type(error).__name__+': '+str(error)));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('smoke','full'),required=True);main(parser.parse_args().mode)
