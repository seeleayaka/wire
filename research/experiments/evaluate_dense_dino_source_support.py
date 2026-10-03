"""Independent dense localization must pass ALL192 burden before holdouts.

No old-teacher proposal/support requirement. New cues must agree across two
overlapping tiles and share the original five-extra budget with current V3.
"""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,BASE,DATA,load,save,read_current_case,read_targets
sys.path.insert(0,str(REPO/'prototype'))
os.environ.update(HF_HUB_OFFLINE='1',YOLO_OFFLINE='True')
from inspection_agent.optional_port_crop_review import sha,read_image
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.port_tiling import tile_windows,near_artificial_edge,box_iou,merge_tiled_ports
from inspection_agent.context_port_recheck import complete
from inspection_agent.teacher_student_port_support import valid
from dense_port_probe import DensePortHead,frozen_features,decode,SOURCE_SCORE,EPOCHS,INPUT
from audit_port_multiscale_acceptance import metric,matches

TRAIN=ROOT/'artifacts/dense_dino_port_probe_20261003'
OUT=TRAIN/'source_acceptance'
POLICY=dict(tile_size=640,stride=480,batch=4,input_size=INPUT,candidate_score=SOURCE_SCORE,
            corroborating_tile_score=.5,agreement_iou=.5,minimum_distinct_tiles=2,
            existing_cues_preserved=True,maximum_primary=5,maximum_extra=5,
            independent_of_old_teacher_proposals=True,automatic_fault_verdict=False,
            same_photo_evidence_not_physical_confirmation=True)


def append_dense(current,raw,digest):
    output=copy.deepcopy(current);output['dense_additions']=[]
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    if len(current['primary'])>5 or remaining<0:raise ValueError('current budget invalid')
    for row in raw['merged_predictions']:
        if len(output['dense_additions'])>=remaining:break
        if not valid(row) or row['confidence']<=SOURCE_SCORE or not complete(row,raw['source_shape']):continue
        votes=sorted({p['source_tile'] for p in raw['edge_kept_predictions'] if valid(p) and p['confidence']>.5 and
                     p['class_id']==row['class_id'] and box_iou(p['box_xyxy'],row['box_xyxy'])>=.5})
        if len(votes)<2:continue
        if any(box_iou(row['box_xyxy'],p['box_xyxy'])>=.5 for p in output['all_predictions']):continue
        selected=copy.deepcopy(row);selected.update(evidence_tier='dense_dino_experimental_manual_review',
            dense_head_sha256=digest,dense_support_tiles=votes,automatic_fault_verdict=False,reference_confirmation_pending=True)
        output['dense_additions'].append(selected);output['all_predictions'].append(selected)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(output['all_predictions'])<=len(current['primary'])+5
    return output


def predict_source(encoder,head,image):
    height,width=image.shape[:2];windows=tile_windows(width,height,size=640,stride=480);rows=[]
    import torch
    with torch.inference_mode():
        for start in range(0,len(windows),4):
            batch=windows[start:start+4];crops=[image[y:b,x:r] for x,y,r,b in batch]
            features=frozen_features(encoder,crops)
            predictions=decode(head(features),[im.shape[:2] for im in crops],score=.5,max_boxes=100)
            for offset,(window,selected) in enumerate(zip(batch,predictions)):
                x,y,r,b=window
                for row in selected:
                    if near_artificial_edge(row['box_xyxy'],window,width,height,16):continue
                    l,t,rr,bb=row['box_xyxy'];rows.append(dict(row,box_xyxy=[l+x,t+y,rr+x,bb+y],source_tile=start+offset))
    return dict(source_shape=[height,width],windows=[list(w) for w in windows],edge_kept_predictions=rows,
                merged_predictions=merge_tiled_ports(rows,.5))


def main():
    if OUT.exists():raise FileExistsError('Preserve evaluation')
    training=load(TRAIN/'full/report.json')
    assert training['status']=='complete' and training['completed_epochs']==EPOCHS and training['qualifies_crop_feasibility']
    assert training['frozen_encoder_unchanged'] and training['nonzero_gradient_steps']>0
    weight=Path(training['checkpoint']['path']);digest=sha(weight);assert digest==training['checkpoint']['sha256']
    encoder_path=REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth'
    assert sha(encoder_path)==training['encoder_sha256']
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==training['runtime_fingerprint']
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('dense_port_probe.py'),
          Path(__file__).with_name('current_port_baseline_audit.py'),Path(__file__).with_name('audit_port_multiscale_acceptance.py'),
          TRAIN/'full/report.json',weight,encoder_path,REPO/'inspection_agent/port_tiling.py')}
    inputs={}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        reportpath=BASE/stage/'report.json';pins[str(reportpath)]=sha(reportpath)
        entries=load(reportpath)['cases'];assert len(entries)==count
        inputs[stage]=[]
        for entry in entries:
            teacher,record=read_current_case(stage,entry,pins)
            inputs[stage].append(dict(entry=entry,teacher=teacher,current=record['trial']))
    OUT.mkdir();save(OUT/'protocol.json',dict(policy=POLICY,pins=pins,runtime_fingerprint=frozen,
        new_detector_sha256=digest,encoder_sha256=training['encoder_sha256'],baseline='Current V3 277/63/33',
        all192_training_then_inner48_then_outer30=True,normal_sources_not_short_circuited_by_old_detector=True,
        exact_short_circuit='All current shared-extra slots full means append impossible; raw dense localization not measured on those images',
        net_train_and_inner_gain_required=True,no_unmatched_increase_no_old_targets_lost_normal_zero=True,
        no_training_or_policy_changes_from_this_scoring=True,validation_reused=True,reference_pending=True,
        no_automatic_deployment=True,field_accuracy=False))
    import torch
    import dino_feature_diff as dino
    torch.set_num_threads(2);encoder=dino._model();encoder.requires_grad_(False).eval();torch.set_num_threads(2)
    saved=torch.load(weight,map_location='cpu',weights_only=True)
    assert saved['input_size']==INPUT and saved['encoder_sha256']==training['encoder_sha256']
    head=DensePortHead();head.load_state_dict(saved['state_dict'],strict=True);head.requires_grad_(False).eval()
    started=time.monotonic()
    try:
        for stage,items in inputs.items():
            folder=OUT/stage;folder.mkdir();records=[];fresh=0;skipped=0
            for index,item in enumerate(items):
                entry,teacher,current=item['entry'],item['teacher'],item['current'];name=entry['image']
                if len(current['all_predictions'])-len(current['primary'])<5:
                    source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
                    raw=predict_source(encoder,head,read_image(source));fresh+=1
                else:
                    raw=dict(source_shape=teacher['predictions']['source_shape'],merged_predictions=[],edge_kept_predictions=[],windows=[]);skipped+=1
                assert raw['source_shape']==teacher['predictions']['source_shape']
                trial=append_dense(current,raw,digest)
                save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,raw=raw,
                     source_sha256=teacher['source_sha256'],dense_head_sha256=digest))
                targets=read_targets(stage,name,raw['source_shape'],entry['label_sha256'],pins)
                old,new=current['all_predictions'],trial['all_predictions'];oh,nh=matches(old,targets)[0],matches(new,targets)[0]
                records.append(dict(image=name,metrics=dict(current=metric(old,targets),trial=metric(new,targets)),
                                    gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['dense_additions'])))
                save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,completed=index+1,total=len(items),
                     fresh_sources=fresh,exact_full_budget_short_circuits=skipped,seconds=round(time.monotonic()-started,2)))
            totals={v:{k:sum(r['metrics'][v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
            normal=sum(r['metrics']['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
            gain=totals['trial']['tp']>totals['current']['tp'] if stage!='outer' else totals['trial']['tp']>=totals['current']['tp']
            passed=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            save(folder/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,
                 fresh_sources=fresh,exact_full_budget_short_circuits=skipped,reference_pending=True,field_accuracy=False,production_changed=False))
            print(str(dict(stage=stage,qualifies=passed,summary=totals)),flush=True)
            if not passed:save(OUT/'progress.json',dict(status='rejected',stage=stage,summary=totals));return
        save(OUT/'progress.json',dict(status='source_only_pass_requires_fresh_reference',no_automatic_deployment=True))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
