"""TRAIN192-only frozen source/reference features; no validation fitting."""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1')
from inspection_agent.optional_port_crop_review import sha,read_image,REFERENCE_SHA
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from paired_port_semantics import paired_features,expected_in_source,valid_boxes,CachedReferenceSIFT
from port_semantic_verifier import embeddings
from audit_port_multiscale_acceptance import overlap
OUT=ROOT/'artifacts/paired_port_semantics_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
OLD=ROOT/'artifacts/port_semantic_verifier_20261003'
PROPOSALS=ROOT/'artifacts/paired_semantic_proposal_potential_20261003'


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)


def cached_alignments():
    mapping={}
    for directory in ('resolution_loose_plug_live_20261003','port_resolution_live_20261003',
        'port_residual_feature_live_20261003','teacher_student_port_live_20261003',
        'main_scenario_live_20261002','consensus_port_live_20261002','context_port_backend_20261002'):
        for p in sorted((ROOT/'artifacts'/directory).rglob('initial_report.json')):
            report=load(p);fp=report.get('image_fingerprints',{});alignment=report.get('alignment',{})
            if (fp.get('reference_sha256')==REFERENCE_SHA and fp.get('stable_during_visual_analysis') is True and
                alignment.get('method')=='automatic_full_frame_sift_homography' and alignment.get('alignment_quality',{}).get('reliable') is True):
                mapping[Path(report['inspection']).name]=(p,report)
    return mapping


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True);args=parser.parse_args()
    out=OUT/('smoke' if args.mode=='smoke' else 'features_train')
    if out.exists():raise FileExistsError('Preserve features')
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json';groups=load(groupspath)
    names=sorted(groups['train_sources']);assert len(names)==192 and set(names).isdisjoint(groups['inner_val_sources'])
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    reference_labels=(DATA/'labels/train01/normal_073.txt').read_text(encoding='utf-8')
    assert not any(int(float(line.split()[0])) in (3,4) for line in reference_labels.splitlines() if line.strip())
    old_samples_path=OLD/'samples.json';old_samples=load(old_samples_path);old_report=load(OLD/'report.json')
    assert old_report['status']=='complete' and old_report['samples']==len(old_samples)
    old_pins=load(OLD/'protocol.json')['pins'];assert {p:sha(p) for p in old_pins}==old_pins
    examples={name:[] for name in names}
    for name in names:
        relevant=[copy.deepcopy(r) for r in old_samples if r['image']==name]
        positives=[r for r in relevant if r['kind']=='gt_port'];others=[r for r in relevant if r['kind']!='gt_port'][:2]
        examples[name]=positives+others
    assert sum(r['kind']=='gt_port' for rows in examples.values() for r in rows)==344
    alignments=cached_alignments()
    if args.mode=='smoke':
        positive_names=[n for n in names if n in alignments and any(r['kind']=='gt_port' for r in examples[n])]
        assert len(positive_names)>=2;names=positive_names[:2]
    else:assert load(OUT/'smoke/report.json')['status']=='complete'
    weight=REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth';digest=sha(weight)
    assert digest=='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
    pins=dict(old_pins)
    for p in (Path(__file__),Path(__file__).with_name('paired_port_semantics.py'),Path(__file__).with_name('port_semantic_verifier.py'),
        groupspath,old_samples_path,OLD/'report.json',referencepath,DATA/'labels/train01/normal_073.txt',weight,
        REPO/'prototype/assembly_auto_review_robust_v3.py',ROOT/'artifacts/paired_port_semantics_preregistration_20261003/PLAN.md'):
        pins[str(p)]=sha(p)
    for name in names:
        for p in (DATA/'images/train01'/name,DATA/'labels/train01'/(Path(name).stem+'.txt'),PROPOSALS/('train_'+Path(name).stem+'_proposals.json')):pins[str(p)]=sha(p)
        if name in alignments:pins[str(alignments[name][0])]=sha(alignments[name][0])
    frozen=resolution_runtime_fingerprint(REPO);out.mkdir(parents=True);started=time.monotonic()
    save(out/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,mode=args.mode,train_sources=names,
        no_inner_or_outer_training=True,encoder_sha256=digest,reference_sha256=REFERENCE_SHA,
        original_SIFT_registration_gates=True,cached_reference_descriptors_only=True,min_context_valid_coverage=.85,
        paired_feature_dimensions=6144,no_metadata_features=True,explicit_alignment_abstentions=True,
        no_label_edits=True,reference_self_pairs_known_normal=True,no_model_deployment=True,field_accuracy=False))
    import cv2
    import numpy as np
    import torch
    import dino_feature_diff as dino
    import assembly_auto_review_robust_v3 as registration
    cv2.setNumThreads(2);torch.set_num_threads(2);encoder=dino._model();encoder.requires_grad_(False).eval();torch.set_num_threads(2)
    reference=read_image(referencepath);metadata=[];features=[];source_records=[];fresh_align=0;reuse_align=0;gt_count=0;cache_pins={}
    all_train_names=sorted(groups['train_sources'])
    def progress(**kw):save(out/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    try:
        with CachedReferenceSIFT(reference) as sift_cache:
            for index,name in enumerate(names):
                progress(phase='alignment',image=name,completed=index,total=len(names),samples=len(metadata))
                image=read_image(DATA/'images/train01'/name);h,w=image.shape[:2];assert [h,w]==[2736,3648]
                source_digest=pins[str(DATA/'images/train01'/name)];gt_rows=[]
                for line in (DATA/'labels/train01'/(Path(name).stem+'.txt')).read_text(encoding='utf-8').splitlines():
                    cls,cx,cy,bw,bh=map(float,line.split())
                    if cls in (3,4):gt_rows.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
                gt_count+=len(gt_rows);fold=all_train_names.index(name)%3
                cached=alignments.get(name)
                if cached:
                    assert cached[1]['image_fingerprints']['source_sha256']==source_digest
                    alignment=copy.deepcopy(cached[1]['alignment']);reuse_align+=1;alignment_provenance=str(cached[0])
                else:
                    cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image);fresh_align+=1;alignment_provenance='fresh_original_mainline_SIFT'
                if not alignment.get('alignment_quality',{}).get('reliable'):
                    record=dict(image=name,fold=fold,status='abstained_registration',alignment=alignment,source_sha256=source_digest,
                        label_sha256=pins[str(DATA/'labels/train01'/(Path(name).stem+'.txt'))],gt_targets=len(gt_rows))
                    source_records.append(record);save(out/(Path(name).stem+'_source.json'),record);continue
                expected,valid=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                sample_rows=copy.deepcopy(examples[name]);candidate_data=load(PROPOSALS/('train_'+Path(name).stem+'_proposals.json'))
                assert candidate_data['image']==name
                for p in candidate_data['candidates']:
                    hit=any(p['class_id']==t['class_id'] and overlap(p['box_xyxy'],t['box'])>=.5 for t in gt_rows)
                    sample_rows.append(dict(box=p['box_xyxy'],label=p['class_id']+1 if hit else 0,kind='novel_weak_proposal',proposal=p))
                indices=valid_boxes([r['box'] for r in sample_rows],valid);selected=[sample_rows[i] for i in indices]
                progress(phase='frozen_paired_crop_embeddings',image=name,completed=index,total=len(names),samples=len(metadata),valid_crops=len(selected))
                boxes=[r['box'] for r in selected];observed=embeddings(encoder,image,boxes);expected_vectors=embeddings(encoder,expected,boxes)
                vectors=paired_features(observed,expected_vectors);records=[];local_features=[]
                for i,row in enumerate(selected):
                    record=dict(row,image=name,fold=fold,source_sha256=source_digest,reference_sha256=REFERENCE_SHA)
                    local_features.append(vectors[i]);records.append(record)
                    if row['kind']=='gt_port':
                        local_features.append(paired_features(expected_vectors[i:i+1],expected_vectors[i:i+1])[0])
                        records.append(dict(image=name,fold=fold,box=row['box'],label=0,kind='reference_self',
                            source_sha256=REFERENCE_SHA,reference_sha256=REFERENCE_SHA))
                array=torch.stack(local_features) if local_features else torch.empty((0,6144))
                assert torch.isfinite(array).all() and not any(p.requires_grad for p in encoder.parameters())
                path=out/(Path(name).stem+'_features.pt');torch.save(dict(features=array,samples=records),path);cache_pins[str(path)]=sha(path)
                features.append(array);metadata.extend(records)
                record=dict(image=name,fold=fold,status='features_ready',alignment=alignment,alignment_provenance=alignment_provenance,
                    source_sha256=source_digest,label_sha256=pins[str(DATA/'labels/train01'/(Path(name).stem+'.txt'))],
                    features=str(path),features_sha256=cache_pins[str(path)],samples=len(records),gt_targets=len(gt_rows),
                    gt_valid=sum(r['kind']=='gt_port' for r in records),novel_proposals=len(candidate_data['candidates']),
                    valid_proposals=sum(r['kind']=='novel_weak_proposal' for r in records),invalid_context_crops=len(sample_rows)-len(selected))
                source_records.append(record);save(out/(Path(name).stem+'_source.json'),record)
                progress(phase='paired_source_complete',image=name,completed=index+1,total=len(names),samples=len(metadata))
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            assert {p:sha(Path(p)) for p in cache_pins}==cache_pins
            combined=torch.cat(features) if features else torch.empty((0,6144));labels=torch.tensor([r['label'] for r in metadata]);folds=torch.tensor([r['fold'] for r in metadata])
            assert len(combined)==len(metadata)
            torch.save(dict(features=combined,labels=labels,folds=folds),out/'features.pt');save(out/'samples.json',metadata)
            report=dict(status='complete',mode=args.mode,samples=len(metadata),gt_targets=gt_count,
                gt_valid=sum(r['kind']=='gt_port' for r in metadata),counts=torch.bincount(labels,minlength=3).tolist(),
                sources=source_records,source_groups=names,fresh_alignments=fresh_align,reused_verified_alignments=reuse_align,
                reference_descriptor_cache_hits=sift_cache.cache_hits,encoder_sha256=digest,frozen_encoder_unchanged=True,
                runtime_fingerprint=frozen,pins=pins,cache_pins=cache_pins,aggregate_feature_sha256=sha(out/'features.pt'),
                samples_sha256=sha(out/'samples.json'),seconds=round(time.monotonic()-started,2),no_validation_training=True,field_accuracy=False)
            save(out/'report.json',report);save(out/'progress.json',dict(status='complete',samples=len(metadata),sources=len(names),seconds=report['seconds']))
            print(str({k:v for k,v in report.items() if k not in ('sources','pins','cache_pins','runtime_fingerprint')}),flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
