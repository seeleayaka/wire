"""Training controls first, then locked-rule retrospective validation."""
import argparse,copy,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/core_port_supplement_20261002'
sys.path.insert(0,str(REPO))
from core_port_supplement_policy import supplement,complete

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def assess(cases, source_split, directory):
    from core_port_resolution_ab_20261002 import DATA,score,sha,ranking
    summary={};groups={};rows=[];missing=[]
    for case in cases:
        prediction=case['predictions'];chosen=supplement(prediction)
        label=DATA/'labels'/source_split/(Path(case['image']).stem+'.txt')
        targets=[];height,width=prediction['source_shape']
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height]))
        all_strong=[p for p in ranking(prediction['merged_predictions'],None) if p['confidence']>.5 and complete(p,prediction['source_shape'])]
        all_above_floor=[p for p in ranking(prediction['merged_predictions'],None) if complete(p,prediction['source_shape'])]
        candidates={'primary':chosen['primary'],'supplementary':chosen['supplementary'],
                    'combined':chosen['all_predictions'],'all_strong_unlimited':all_strong,'all_above_floor_unlimited':all_above_floor}
        metrics={stage:score(preds,targets) for stage,preds in candidates.items()}
        kind=case['image'].split('_')[0]
        for stage,metric in metrics.items():
            for aggregate in (summary.setdefault(stage,{k:0 for k in metric}),groups.setdefault(kind,{}).setdefault(stage,{k:0 for k in metric})):
                for key,value in metric.items():aggregate[key]+=value
        rows.append(dict(image=case['image'],metrics=metrics,label_sha256=sha(label)))
        # Per-target cause diagnosis is descriptive, not a rule-selection input.
        for target in targets:
            from core_port_precision_policy import iou
            best=lambda selected: max((iou(target['box'],p['box_xyxy']) for p in selected if target['class_id']==p['class_id']),default=0)
            if best(chosen['primary'])>=.5:continue
            cause=('above_threshold_but_budget_truncated' if best(all_strong)>=.5 else
                   'score_0.25_to_0.5' if best(all_above_floor)>=.5 else 'no_precise_same_class_prediction_above_0.25')
            missing.append(dict(image=case['image'],class_id=target['class_id'],target_box=target['box'],cause=cause,
                recovered_by_supplement=best(chosen['all_predictions'])>=.5))
        save(directory/(Path(case['image']).stem+'_evaluation.json'),dict(metrics=metrics,selected=chosen))
    a,b=summary['primary'],summary['combined']
    report=dict(status='complete',summary=summary,groups=groups,cases=rows,misses=missing,
        qualifies=b['tp']>a['tp'] and b['unmatched']<=a['unmatched'],budget_change_explicit=True,
        primary_limit=5,supplementary_limit=5,field_accuracy_claimed=False,
        scope='same-camera port localization only, labels do not certify physical faults')
    save(directory/'report.json',report)
    print(json.dumps(dict(summary=summary,qualifies=report['qualifies']),indent=2),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','inner','outer'],required=True);args=parser.parse_args()
    target=OUT/args.mode
    if target.exists():raise FileExistsError('Fresh output required')
    (target/'config/Ultralytics').mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(target/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
    import torch
    from core_port_resolution_ab_20261002 import DATA,WEIGHT,MANIFEST
    from inspection_agent.optional_port_crop_review import CONFIG,sha,read_image,predict
    os.environ['YOLO_CONFIG_DIR']=str(target/'config')
    manifest=load(MANIFEST);config=copy.deepcopy(CONFIG)
    weight_hash=sha(WEIGHT);assert weight_hash==load(REPO/'config/port_crop_calibration_frozen_20260930.json')['fingerprints']['weight']
    if args.mode=='train':
        if (OUT/'protocol.json').exists():raise FileExistsError('Do not overwrite protocol')
        names=sorted({r['source_image'] for r in manifest['records'] if r['split']=='train'})
        chosen=[]
        for kind in ('disconnected','normal','damaged','misrouted'):chosen+=sorted(n for n in names if n.startswith(kind+'_'))[:8]
        assert len(chosen)==32
        save(OUT/'protocol.json',dict(training_controls=chosen,primary='existing high-score complete-frame max5',
            additional_maximum=5,additional_score=.75,minimum_strong_distinct_tiles=2,strong_vote_threshold=.5,
            same_class_iou=.5,frame_margin=16,reference_and_roi_gates_required_for_deployment=True,
            added_review_burden_explicit=True,no_image_names_or_scene_coordinates_in_selector=True,
            no_val_or_test_labels_for_rule_design=True,training=False,weight_sha256=weight_hash,config=config))
        from ultralytics import YOLO
        torch.set_num_threads(4);torch.manual_seed(20260929)
        model=YOLO(str(WEIGHT));cases=[];start=time.monotonic()
        for name in chosen:
            path=DATA/'images/train01'/name
            case=dict(image=name,predictions=predict(model,read_image(path)),source_sha256=sha(path),weight_sha256=weight_hash)
            save(target/(Path(name).stem+'_predictions.json'),case);cases.append(case)
            save(target/'progress.json',dict(completed=len(cases),total=len(chosen),seconds=round(time.monotonic()-start,2)))
            print(f'{len(cases)}/{len(chosen)} {name}',flush=True)
        assert CONFIG==config and sha(WEIGHT)==weight_hash
        assess(cases,'train01',target)
    else:
        assert load(OUT/'train/report.json')['qualifies'],'Training control gate failed'
        if args.mode=='outer':assert load(OUT/'inner/report.json')['qualifies'],'Inner validation gate failed'
        previous=ROOT/('artifacts/core_precision_full_'+args.mode+'_20261002')
        source=load(previous/'report.json');cases=[]
        for row in source['cases']:
            case=load(previous/(Path(row['image']).stem+'_predictions.json'))
            path=DATA/'images'/('train01' if args.mode=='inner' else 'val01')/case['image']
            assert sha(path)==case['source_sha256'] and case['weight_sha256']==weight_hash
            cases.append(case)
        report=assess(cases,'train01' if args.mode=='inner' else 'val01',target)
        report.update(new_inference=False,already_inspected_holdout=True)
        save(target/'report.json',report)

if __name__=='__main__':main()
