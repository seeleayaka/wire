"""Single fixed no-training experiment; source gain gate before validation."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,cached_alignments,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case,read_targets
from audit_port_multiscale_acceptance import matches,metric
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import valid_boxes,expected_in_source,paired_features,embeddings
from paired_port_semantics import CachedReferenceSIFT
from raw_pose_consensus import proposals,select
OUT=ROOT/'artifacts/raw_pose_consensus_20261004'
PLAN=ROOT/'artifacts/raw_pose_consensus_preregistration_20261004/PLAN.md'
CURRENT=ROOT/'artifacts/paired_pose_native_three_20261004'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections'

def main():
    if OUT.exists():raise FileExistsError('Preserve finite raw consensus trial')
    import torch
    import cv2
    import dino_feature_diff as dino
    import assembly_auto_review_robust_v3 as registration
    import psutil
    assert psutil.virtual_memory().available>6*2**30,'No competing memory-heavy SAM job'
    torch.set_num_threads(2);cv2.setNumThreads(1)
    frozen=native_pose_runtime_fingerprint(REPO)
    headpath=REPO/HEAD_RELATIVE;assert sha(headpath)==HEAD_SHA
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('raw_pose_consensus.py'),PLAN,headpath,referencepath,
        REPO/'inspection_agent/paired_native_pose_features.py',REPO/'inspection_agent/paired_port_features.py',
        REPO/'inspection_agent/paired_port_median_features.py',REPO/'prototype/assembly_auto_review_robust_v3.py')}
    OUT.mkdir();started=time.monotonic();stages={};encoder=None
    def progress(**kw):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,no_training=True,fixed_p98_three_votes=True,
        current_baseline=[295,68,40],default_off_unchanged=True,field_accuracy=False))
    head=torch.nn.Linear(6144,3);checkpoint=torch.load(headpath,map_location='cpu',weights_only=True)
    head.load_state_dict(checkpoint['state_dict']);head.eval().requires_grad_(False)
    reference=read_image(referencepath);alignments=cached_alignments()
    try:
        with CachedReferenceSIFT(reference):
            for stage,expected in (('train',192),('inner',48),('outer',30)):
                folder=OUT/stage;folder.mkdir();indexpath=PREP/stage/'index.json';pins[str(indexpath)]=sha(indexpath)
                indexed={r['image']:r for r in load(indexpath)['records']}
                entries=load(BASE/stage/'report.json')['cases'];assert len(entries)==expected
                rows=[];proposal_count=0;normal=0
                # Complete proposal construction before any stage labels are read.
                prepared=[]
                for entry in entries:
                    name=entry['image'];stem=Path(name).stem;path=Path(indexed[name]['path'])
                    assert sha(path)==indexed[name]['sha256'];pins[str(path)]=sha(path);paired=load(path)
                    teacher,old=read_current_case(stage,entry,pins);assert teacher==paired['teacher']
                    currentpath=CURRENT/('full_train' if stage=='train' else stage)/(stem+'_predictions.json')
                    pins[str(currentpath)]=sha(currentpath);current=load(currentpath)['trial']
                    remaining=5-(len(current['all_predictions'])-len(current['primary']));assert remaining>=0
                    candidates=proposals(teacher,[teacher,paired['student'],paired['feature'],old['alternative']],current) if remaining else []
                    prepared.append((entry,current,candidates))
                for i,(entry,current,candidates) in enumerate(prepared):
                    name=entry['image'];stem=Path(name).stem;scores=[];alignment=None
                    progress(stage=stage,image=name,completed=i,total=expected,proposals=proposal_count,phase='frozen_crop_features')
                    source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;pins[str(source)]=sha(source)
                    if candidates:
                        image=read_image(source)
                        if name in alignments:
                            provenance,cached=alignments[name];pins[str(provenance)]=sha(provenance)
                            assert cached['image_fingerprints']['source_sha256']==pins[str(source)];alignment=cached['alignment']
                        else:cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                        if alignment.get('alignment_quality',{}).get('reliable'):
                            expected_image,mask=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                            candidates=[candidates[j] for j in valid_boxes([r['box_xyxy'] for r in candidates],mask)]
                            if candidates:
                                if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                                boxes=[r['box_xyxy'] for r in candidates]
                                vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected_image,boxes))
                                with torch.inference_mode():scores=head(vectors).softmax(dim=1).tolist()
                        else:candidates=[]
                    trial=select(current,candidates,scores,HEAD_SHA);proposal_count+=len(candidates)
                    assert trial['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
                    save(folder/(stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=candidates,probabilities=scores,alignment=alignment))
                    targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                    oldhits=matches(current['all_predictions'],targets)[0];newhits=matches(trial['all_predictions'],targets)[0]
                    rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),lost=sorted(oldhits-newhits),gained=sorted(newhits-oldhits)))
                    if name.startswith('normal_'):normal+=len(trial['all_predictions'])
                totals={version:{k:sum(r[version][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
                qualifies=(totals['trial']['tp']>totals['current']['tp'] if stage!='outer' else totals['trial']['tp']>=totals['current']['tp']) and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in rows) and normal==0
                stages[stage]=dict(qualifies=qualifies,summary=totals,valid_proposals=proposal_count,normal_cues=normal)
                save(folder/'report.json',dict(status='complete',**stages[stage],cases=rows));print(dict(stage=stage,**stages[stage]),flush=True)
                if not qualifies:break
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        failed=next((name for name,value in stages.items() if not value['qualifies']),None)
        result=dict(status='rejected' if failed else 'source_pass_requires_actual_reference_ROI_gates',failed_stage=failed,stages=stages,pins=pins,
            runtime=frozen,no_deployment=True,no_training=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',result)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise

if __name__=='__main__':main()
