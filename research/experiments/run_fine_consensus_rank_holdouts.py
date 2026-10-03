"""Fresh fixed INNER then conditional OUTER, no fitting or deployment."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from inspection_agent.optional_port_crop_review import CONFIG
from inspection_agent.teacher_student_port_support import TEACHER_RELATIVE,TEACHER_SHA,STUDENT_RELATIVE,STUDENT_SHA
from paired_port_semantics import CachedReferenceSIFT
from raw_pose_consensus import proposals
from consensus_rank import select
from fine_voter_quality import attach
from audit_port_multiscale_acceptance import metric,matches
OUT=ROOT/'artifacts/fine_consensus_rank_holdouts_20261004'
SOURCE=ROOT/'artifacts/fine_consensus_rank_20261004'
AUDIT=ROOT/'artifacts/fine_consensus_rank_replay_20261004/report.json'
CURRENT=ROOT/'artifacts/paired_pose_native_three_20261004'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections'
PLAN=ROOT/'artifacts/fine_consensus_rank_holdouts_preregistration_20261004/PLAN.md'


def main():
    import psutil
    import cv2
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if OUT.exists():raise FileExistsError('Preserve fixed held-out run')
    assert psutil.virtual_memory().available>6*2**30,'Do not compete for inference memory'
    source=load(SOURCE/'report.json');audit=load(AUDIT)
    assert source['status']=='source_pass_requires_fresh_fine_holdouts_and_actual_gates' and source['stages']['train']['qualifies']
    assert audit['status']=='pass' and audit['source_report_sha256']==sha(SOURCE/'report.json')
    assert all(sha(Path(p))==v for p,v in source['pins'].items())
    frozen=native_pose_runtime_fingerprint(REPO);assert frozen==source['runtime']
    accepted=ROOT/'artifacts/paired_pose_native_three_audit_20261004/report.json';assert load(accepted)['status']=='complete'
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,SOURCE/'report.json',AUDIT,accepted,CURRENT/'report.json',
        referencepath,REPO/HEAD_RELATIVE,Path(__file__).with_name('paired_fine_tile_views.py'),
        Path(__file__).with_name('fine_voter_quality.py'),Path(__file__).with_name('consensus_rank.py'),
        Path(__file__).with_name('novel_box_geometry.py'),Path(__file__).with_name('relative_port_box.py'),
        Path(__file__).with_name('raw_pose_consensus.py'),REPO/'inspection_agent/paired_native_pose_features.py',REPO/'inspection_agent/paired_port_features.py')}
    assert pins[str(REPO/HEAD_RELATIVE)]==HEAD_SHA
    OUT.mkdir();configdir=OUT/'config';(configdir/'Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',configdir/'Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(configdir),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    from ultralytics import YOLO
    from paired_fine_tile_views import infer
    import assembly_auto_review_robust_v3 as registration
    weights=[REPO/TEACHER_RELATIVE,REPO/STUDENT_RELATIVE];digests=[TEACHER_SHA,STUDENT_SHA]
    assert [sha(p) for p in weights]==digests;pins.update({str(p):v for p,v in zip(weights,digests)})
    models=[YOLO(str(p)) for p in weights]
    assert all(m.task=='segment' and dict(m.names)=={0:'unplugged_plug',1:'unplugged_jack'} for m in models)
    class Capped:
        def __init__(self,model):self.model=model
        def predict(self,*a,**kw):
            torch.set_num_threads(2);result=self.model.predict(*a,**kw);torch.set_num_threads(2);return result
    wrapped=[Capped(m) for m in models]
    head=torch.nn.Linear(6144,3);head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    reference=read_image(referencepath);encoder=None;started=time.monotonic();stages={};original_config=copy.deepcopy(CONFIG)
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,source_report_sha256=sha(SOURCE/'report.json'),
        tile_size=960,tile_stride=720,input_size=960,probability_gate=.98,distinct_SHA_votes=3,duplicate_IoMin=.5,
        geometric_rank='median_best_IoU_per_distinct_voter',original_config=original_config,no_training=True,no_deployment=True,field_accuracy=False))
    progress=lambda **kw:save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    def finish(failed):
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen and CONFIG==original_config
        result=dict(status='rejected' if failed else 'holdout_pass_requires_actual_reference_ROI_and_Qt_SAM',failed_stage=failed,stages=stages,
            pins=pins,runtime=frozen,no_training=True,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],seconds=result['seconds']));print(dict(status=result['status'],stages=stages),flush=True)
    try:
        with CachedReferenceSIFT(reference):
            for stage,count,base_tp,base_fp in [('inner',48,68,0),('outer',30,40,1)]:
                folder=OUT/stage;folder.mkdir();indexpath=PREP/stage/'index.json';pins[str(indexpath)]=sha(indexpath)
                indexed={r['image']:r for r in load(indexpath)['records']};entries=load(BASE/stage/'report.json')['cases'];assert len(entries)==count
                rows=[];inferred=0;abstained=0
                for completed,entry in enumerate(entries):
                    name=entry['image'];progress(stage=stage,phase='fresh_fine_and_original_pair_semantics',image=name,completed=completed,total=count,inferred=inferred)
                    pairpath=Path(indexed[name]['path']);assert sha(pairpath)==indexed[name]['sha256'];pins[str(pairpath)]=sha(pairpath);pair=load(pairpath)
                    teacher,old=read_current_case(stage,entry,pins);assert teacher==pair['teacher']
                    currentpath=CURRENT/stage/(Path(name).stem+'_predictions.json');pins[str(currentpath)]=sha(currentpath);accepted_case=load(currentpath)
                    assert accepted_case['head_sha256']==HEAD_SHA;current=accepted_case['trial'];pool=[teacher,pair['student'],pair['feature'],old['alternative']]
                    sourcepath=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;source_sha=sha(sourcepath);pins[str(sourcepath)]=source_sha
                    remaining=5-(len(current['all_predictions'])-len(current['primary']));assert remaining>=0
                    eligible=remaining>0 and any(p['confidence']>.05 for model in pool for p in model['predictions']['merged_predictions'])
                    new_views=[];native=[];scores=[];alignment=None;reason=None
                    if eligible:
                        image=read_image(sourcepath);assert image.shape[:2]==(2736,3648)
                        cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                        if alignment.get('alignment_quality',{}).get('reliable'):
                            expected,mask=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                            for model,weight in zip(wrapped,digests):
                                predictions=infer(model,image)
                                new_views.append(dict(image=name,source_sha256=source_sha,weight_sha256=weight,predictions=predictions,view='fresh_fine960_stride720'))
                            inferred+=1;native=proposals(teacher,pool+new_views,current)
                            native=[native[i] for i in valid_boxes([p['box_xyxy'] for p in native],mask)]
                            native=attach(native,pool+new_views,source_sha,image.shape[:2])
                            if native:
                                import dino_feature_diff as dino
                                if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                                boxes=[p['box_xyxy'] for p in native];vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                                with torch.inference_mode():scores=head(vectors).softmax(1).tolist()
                        else:abstained+=1;reason='unreliable_original_SIFT'
                    else:reason='budget_full_or_no_existing_seed_evidence'
                    trial=select(current,native,scores,HEAD_SHA)
                    save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=native,probabilities=scores,
                        new_views=new_views,alignment=alignment,eligible=eligible,skip_reason=reason,head_sha256=HEAD_SHA,source_sha256=source_sha))
                    targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                    a=matches(current['all_predictions'],targets)[0];b=matches(trial['all_predictions'],targets)[0]
                    rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(b-a),lost=sorted(a-b),skip_reason=reason))
                totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
                assert totals['current']['tp']==base_tp and totals['current']['unmatched']==base_fp
                normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
                qualifies=(totals['trial']['tp']>base_tp if stage=='inner' else totals['trial']['tp']>=base_tp) and totals['trial']['unmatched']<=base_fp and normal==0 and not any(r['lost'] for r in rows)
                summary=dict(qualifies=qualifies,summary=totals,normal_cues=normal,fresh_fine_sources=inferred,registration_abstentions=abstained)
                stages[stage]=summary;save(folder/'report.json',dict(status='complete',**summary,cases=rows));print(dict(stage=stage,**summary),flush=True)
                if not qualifies:finish(stage);return
        finish(None)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise


if __name__=='__main__':main()
