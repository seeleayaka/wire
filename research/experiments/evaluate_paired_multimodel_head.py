"""Exact accepted paired head on extra distinct-checkpoint weak witnesses."""
import copy,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,OUT as ORIGINAL,load,save,sha,read_image,cached_alignments
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE,read_targets
from paired_shape_features import crop_signature
from paired_port_semantics import expected_in_source,valid_boxes,paired_features,CachedReferenceSIFT
from port_semantic_verifier import embeddings
from paired_port_semantic_selection import select
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint,HEAD_SHA
from audit_port_multiscale_acceptance import metric,matches
PROPOSALS=ROOT/'artifacts/paired_multimodel_proposal_ceiling_20261003'
OUT=ROOT/'artifacts/paired_multimodel_head_20261003'


def key(box):return tuple(crop_signature(box,s) for s in (1.5,3.))


def main():
    if OUT.exists():raise FileExistsError('Preserve expanded candidate evaluation')
    import torch,cv2,numpy as np
    torch.set_num_threads(1);cv2.setNumThreads(1)
    frozen=paired_runtime_fingerprint(REPO);prepared=load(GEOMETRY/'features_train/report.json');weight=REPO/'output/paired_port_geometry_20261003/last_head.pt'
    assert sha(weight)==HEAD_SHA
    source_meta=load(GEOMETRY/'features_train/samples.json');assert sha(GEOMETRY/'features_train/samples.json')==prepared['samples_sha256']
    assert sha(GEOMETRY/'features_train/features.pt')==prepared['aggregate_feature_sha256']
    vectors=torch.load(GEOMETRY/'features_train/features.pt',map_location='cpu',weights_only=True)['features'];cache={}
    for i,row in enumerate(source_meta):
        if row['kind']=='reference_self':continue
        cache.setdefault((row['image'],key(row['box'])),vectors[i])
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_port_semantics.py'),Path(__file__).with_name('port_semantic_verifier.py'),
        Path(__file__).with_name('paired_shape_features.py'),Path(__file__).with_name('paired_port_semantic_selection.py'),
        ROOT/'artifacts/paired_multimodel_head_preregistration_20261003/PLAN.md',PROPOSALS/'report.json',weight,
        GEOMETRY/'features_train/samples.json',GEOMETRY/'features_train/features.pt',GEOMETRY/'features_train/report.json')}
    for p,digest in load(PROPOSALS/'report.json')['pins'].items():assert sha(Path(p))==digest;pins[p]=digest
    OUT.mkdir();started=time.monotonic()
    save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,head_sha256=HEAD_SHA,no_training=True,
        original_exact_signature_train_feature_reuse=True,no_reference_self_feature_reuse=True,
        fresh_other_embeddings=True,GT_after_final_source_selection=True,threads=1,no_deployment=True,field_accuracy=False))
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True);assert checkpoint['input_dimensions']==6144
    head=torch.nn.Linear(6144,3);head.load_state_dict(checkpoint['state_dict']);head.eval().requires_grad_(False)
    encoder=None;reference=read_image(DATA/'images/train01/normal_073.JPG');alignments=cached_alignments();summaries={}
    import assembly_auto_review_robust_v3 as registration
    try:
        with CachedReferenceSIFT(reference):
            for stage,count in (('train',192),('inner',48),('outer',30)):
                folder=OUT/stage;folder.mkdir();entries=load(BASE/stage/'report.json')['cases'];assert len(entries)==count;records=[]
                for index,entry in enumerate(entries):
                    name=entry['image'];path=PROPOSALS/(stage+'_'+Path(name).stem+'_proposals.json');pins[str(path)]=sha(path);record=load(path)
                    current=record['current'];native=record['extended_candidates'];valid_native=[];scores=[];alignment=None;fresh=0;reused=0;status='no_candidate_exact_short_circuit'
                    save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,completed=index,total=count,
                        phase='paired_extended_candidates',seconds=round(time.monotonic()-started,2)))
                    if native:
                        source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;pins[str(source)]=sha(source);image=read_image(source)
                        if stage=='train':
                            sp=ORIGINAL/'features_train'/(Path(name).stem+'_source.json');pins[str(sp)]=sha(sp);sr=load(sp)
                            assert sr['source_sha256']==pins[str(source)];alignment=copy.deepcopy(sr['alignment'])
                        elif name in alignments:
                            sp,sr=alignments[name];pins[str(sp)]=sha(sp);assert sr['image_fingerprints']['source_sha256']==pins[str(source)]
                            alignment=copy.deepcopy(sr['alignment'])
                        else:cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                        if alignment.get('alignment_quality',{}).get('reliable'):
                            expected,mask=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                            valid_native=[native[i] for i in valid_boxes([r['box_xyxy'] for r in native],mask)]
                            values=[];missing=[];positions=[]
                            for j,row in enumerate(valid_native):
                                cached=cache.get((name,key(row['box_xyxy']))) if stage=='train' else None
                                if cached is None:values.append(None);missing.append(row['box_xyxy']);positions.append(j)
                                else:values.append(cached);reused+=1
                            if missing:
                                if encoder is None:
                                    import dino_feature_diff as dino
                                    encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(1)
                                array=paired_features(embeddings(encoder,image,missing),embeddings(encoder,expected,missing));fresh=len(missing)
                                for pos,vector in zip(positions,array):values[pos]=vector
                            with torch.inference_mode():
                                array=torch.stack(values) if values else torch.empty((0,6144));scores=head(array).softmax(dim=1).tolist()
                            status='paired_inferred' if values else 'invalid_context_abstention'
                        else:status='registration_abstention'
                    trial=select(current,valid_native,scores,HEAD_SHA)
                    save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=valid_native,
                        probabilities=scores,alignment=alignment,feature_status=status,fresh_pairs=fresh,reused_pairs=reused))
                    targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins);oh,nh=matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
                    records.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
                        gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_semantic_additions']),fresh_pairs=fresh,reused_pairs=reused,feature_status=status))
                totals={kind:{k:sum(r[kind][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for kind in ('current','trial')}
                accepted_folder=GEOMETRY/'full_train' if stage=='train' else GEOMETRY/'holdouts'/stage
                assert totals['current']==load(accepted_folder/'report.json')['summary']['trial']
                normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
                gain=totals['trial']['tp']>totals['current']['tp'] if stage!='outer' else totals['trial']['tp']>=totals['current']['tp']
                qualifies=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
                assert {p:sha(Path(p)) for p in pins}==pins and paired_runtime_fingerprint(REPO)==frozen
                result=dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,
                    fresh_pairs=sum(r['fresh_pairs'] for r in records),reused_pairs=sum(r['reused_pairs'] for r in records),
                    fixed_full_head_not_OOF=True,no_training=True,no_deployment=True,field_accuracy=False)
                save(folder/'report.json',result);summaries[stage]=dict(qualifies=qualifies,summary=totals);print(str(dict(stage=stage,**summaries[stage])),flush=True)
                if not qualifies:
                    save(OUT/'report.json',dict(status='rejected',stage=stage,stages=summaries,pins=pins,no_deployment=True,seconds=round(time.monotonic()-started,2)))
                    save(OUT/'progress.json',dict(status='rejected',stage=stage));return
        save(OUT/'report.json',dict(status='source_pass_requires_reference_ROI_SAM_Qt',stages=summaries,pins=pins,no_deployment=True,seconds=round(time.monotonic()-started,2)))
        save(OUT/'progress.json',dict(status='complete',stages=summaries,no_deployment=True))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise

if __name__=='__main__':main()
