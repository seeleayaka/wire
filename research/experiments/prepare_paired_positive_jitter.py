"""Preserve old3134 vectors; fresh uniform positive jitter and reference-self."""
import argparse,copy,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,OUT as ORIGINAL,load,save,sha,read_image
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from paired_positive_jitter_examples import positive_jitters
from paired_semantic_geometry_examples import training_label
from paired_port_semantics import expected_in_source,valid_boxes,paired_features
from port_semantic_verifier import embeddings
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
OUT=ROOT/'artifacts/paired_positive_jitter_20261003'


def main(mode):
    destination=OUT/('smoke' if mode=='smoke' else 'features_train')
    if destination.exists():raise FileExistsError('Preserve positive-jitter features')
    import torch,cv2
    prepared=load(GEOMETRY/'features_train/report.json');meta=load(GEOMETRY/'features_train/samples.json')
    assert prepared['samples']==3134 and len(meta)==3134 and sum(r['kind']=='gt_port' for r in meta)==344
    assert sha(GEOMETRY/'features_train/features.pt')==prepared['aggregate_feature_sha256']
    assert sha(GEOMETRY/'features_train/samples.json')==prepared['samples_sha256']
    data=torch.load(GEOMETRY/'features_train/features.pt',map_location='cpu',weights_only=True);groups=prepared['source_groups']
    if mode=='smoke':groups=[name for name in groups if any(r['image']==name and r['kind']=='gt_port' for r in meta)][:2]
    else:assert load(OUT/'smoke/report.json')['fresh_jitter_inference_passed']
    frozen=paired_runtime_fingerprint(REPO);oldruntime=resolution_runtime_fingerprint(REPO)
    assert prepared['runtime_fingerprint']==oldruntime
    pins=dict(prepared['pins'])
    for path in (Path(__file__),Path(__file__).with_name('paired_positive_jitter_examples.py'),Path(__file__).with_name('paired_semantic_geometry_examples.py'),
        Path(__file__).with_name('port_semantic_verifier.py'),GEOMETRY/'features_train/report.json',GEOMETRY/'features_train/features.pt',
        GEOMETRY/'features_train/samples.json',ROOT/'artifacts/paired_positive_jitter_preregistration_20261003/PLAN.md'):
        pins[str(path)]=sha(path)
    for name in groups:
        path=ORIGINAL/'features_train'/(Path(name).stem+'_source.json');pins[str(path)]=sha(path)
    assert {p:sha(Path(p)) for p in pins}==pins
    destination.mkdir(parents=True);save(destination/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,train_sources=groups,
        mode=mode,shifts15_percent=True,scales=[.8,1.2],label_against_all_original_GT=True,reference_self_for_every_valid_jitter=True,
        feature_dimensions=6144,no_metadata_features=True,encoder_frozen=True,no_deployment=True,field_accuracy=False))
    import dino_feature_diff as dino
    cv2.setNumThreads(2);torch.set_num_threads(2);model=dino._model();model.eval().requires_grad_(False);torch.set_num_threads(2)
    reference=read_image(DATA/'images/train01/normal_073.JPG');started=time.monotonic()
    old_indices=[i for i,r in enumerate(meta) if r['image'] in groups];values=[data['features'][old_indices]];metadata=[copy.deepcopy(meta[i]) for i in old_indices]
    records=[];cache_pins={};added=0;abstained=0
    try:
        for index,name in enumerate(groups):
            source_path=ORIGINAL/'features_train'/(Path(name).stem+'_source.json');source=load(source_path)
            gt=[r for r in meta if r['image']==name and r['kind']=='gt_port'];truth=[dict(class_id=r['label']-1,box=r['box']) for r in gt]
            save(destination/'progress.json',dict(status='running',pid=os.getpid(),image=name,completed=index,total=len(groups),samples=len(metadata),
                phase='fresh_jitter_and_reference_self',seconds=round(time.monotonic()-started,2)))
            rows=[]
            if gt:
                imagepath=DATA/'images/train01'/name;assert sha(imagepath)==source['source_sha256'];image=read_image(imagepath)
                assert source['alignment']['alignment_quality']['reliable'];expected,mask=expected_in_source(reference,source['alignment']['source_to_reference_homography'],image.shape[:2])
                for target in gt:
                    for box in positive_jitters(target['box']):
                        label,iou=training_label(box,truth)
                        rows.append(dict(image=name,fold=target['fold'],box=box,label=label,kind='synthetic_near_correct_jitter',
                            training_GT_maximum_iou=iou,source_sha256=source['source_sha256']))
                valid=valid_boxes([r['box'] for r in rows],mask);abstained+=len(rows)-len(valid);selected=[rows[i] for i in valid];boxes=[r['box'] for r in selected]
                observed=embeddings(model,image,boxes);expected_vectors=embeddings(model,expected,boxes)
                paired=paired_features(observed,expected_vectors);normal=paired_features(expected_vectors,expected_vectors)
                selves=[dict(image=name,fold=r['fold'],box=r['box'],label=0,kind='jitter_reference_self',source_sha256=source['source_sha256'],
                    observed_is_expected_reference=True) for r in selected]
                array=torch.cat((paired,normal));samples=selected+selves
                assert torch.isfinite(array).all();added+=len(selected)
            else:array=torch.empty((0,6144));samples=[]
            path=destination/(Path(name).stem+'_new_features.pt');torch.save(dict(features=array,samples=samples),path);cache_pins[str(path)]=sha(path)
            values.append(array);metadata.extend(samples);row=copy.deepcopy(source)
            row.update(jitter_examples=len(samples)//2,jitter_positive=sum(r['label']>0 for r in samples),jitter_reference_self=len(samples)//2)
            records.append(row);save(destination/(Path(name).stem+'_source.json'),row)
        combined=torch.cat(values);labels=torch.tensor([r['label'] for r in metadata]);folds=torch.tensor([r['fold'] for r in metadata])
        assert combined.shape==(len(metadata),6144) and torch.isfinite(combined).all()
        assert {p:sha(Path(p)) for p in pins}==pins and {p:sha(Path(p)) for p in cache_pins}==cache_pins
        assert paired_runtime_fingerprint(REPO)==frozen and not any(p.requires_grad for p in model.parameters())
        torch.save(dict(features=combined,labels=labels,folds=folds),destination/'features.pt');save(destination/'samples.json',metadata)
        result=dict(status='complete',mode=mode,samples=len(metadata),gt_targets=sum(r['kind']=='gt_port' for r in metadata),gt_valid=sum(r['kind']=='gt_port' for r in metadata),
            counts=torch.bincount(labels,minlength=3).tolist(),source_groups=groups,sources=records,pins=pins,cache_pins=cache_pins,
            aggregate_feature_sha256=sha(destination/'features.pt'),samples_sha256=sha(destination/'samples.json'),runtime_fingerprint=oldruntime,
            accepted_paired_runtime=frozen,encoder_sha256=prepared['encoder_sha256'],frozen_encoder_unchanged=True,no_validation_training=True,
            fresh_jitter_inference_passed=True,added_jitter_pairs=added,invalid_context_abstentions=abstained,
            old_samples_preserved=len(old_indices),seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False)
        save(destination/'report.json',result);save(destination/'progress.json',dict(status='complete',samples=len(metadata),seconds=result['seconds']))
        print(str({k:v for k,v in result.items() if k not in ('sources','pins','cache_pins','source_groups','runtime_fingerprint','accepted_paired_runtime')}),flush=True)
    except BaseException as error:
        save(destination/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('smoke','full'),required=True);main(parser.parse_args().mode)
