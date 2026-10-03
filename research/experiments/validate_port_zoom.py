"""Fresh proposal-crop inference, frozen before labels; retained base caches checked."""
import argparse, copy, json, os, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/core_port_zoom_20261002'
sys.path.insert(0,str(REPO))
from core_port_zoom_policy import POLICY,proposals,windows,select_zoom
from core_port_supplement_policy import supplement

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def evaluate(cases,split,directory,modes):
    from core_port_resolution_ab_20261002 import DATA,score,sha
    totals={};groups={};rows=[]
    for case in cases:
        height,width=case['predictions']['source_shape'];targets=[]
        label=DATA/'labels'/split/(Path(case['image']).stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height]))
        selected={'baseline':supplement(case['predictions'])}
        selected.update({m:select_zoom(case,m) for m in modes})
        metrics={stage:score(s['all_predictions'],targets) for stage,s in selected.items()}
        kind=case['image'].split('_')[0]
        for stage,metric in metrics.items():
            for aggregate in (totals.setdefault(stage,{k:0 for k in metric}),groups.setdefault(kind,{}).setdefault(stage,{k:0 for k in metric})):
                for key,value in metric.items():aggregate[key]+=value
        save(directory/(Path(case['image']).stem+'_evaluation.json'),dict(selected=selected,metrics=metrics,label_sha256=sha(label)))
        rows.append(dict(image=case['image'],metrics=metrics,proposals=len(case['zoom_evidence']),seconds=case['zoom_seconds']))
    qualifies={m:totals[m]['tp']>totals['baseline']['tp'] and totals[m]['unmatched']<=totals['baseline']['unmatched']
        and groups['normal'][m]['predictions']<=groups['normal']['baseline']['predictions'] for m in modes}
    report=dict(status='complete',summary=totals,groups=groups,qualifies=qualifies,cases=rows,
        fresh_crop_inference=True,base_inference_reused=True,training=False,field_accuracy_claimed=False,
        scope='Same-camera class-specific port IoU>=0.5 localization; not physical fault/connection validation')
    save(directory/'report.json',report);print(json.dumps(dict(summary=totals,qualifies=qualifies)),flush=True)
    return report

def main():
    global OUT,POLICY,proposals,windows,select_zoom
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','inner','outer'],required=True)
    parser.add_argument('--strategy',choices=['zoom','context','recenter','recheck'],default='zoom')
    args=parser.parse_args();target=OUT/args.mode
    policy_file=Path(__file__).with_name('core_port_zoom_policy.py')
    dependency_file=policy_file
    if args.strategy=='context':
        from core_port_context_policy import POLICY as context_policy,proposals as context_proposals,windows as context_windows,select_zoom as context_select
        POLICY,proposals,windows,select_zoom=context_policy,context_proposals,context_windows,context_select
        OUT=ROOT/'artifacts/core_port_context_20261002';target=OUT/args.mode
        policy_file=Path(__file__).with_name('core_port_context_policy.py')
    elif args.strategy=='recenter':
        from core_port_recenter_policy import POLICY as recenter_policy,proposals as recenter_proposals,windows as recenter_windows,select_zoom as recenter_select
        POLICY,proposals,windows,select_zoom=recenter_policy,recenter_proposals,recenter_windows,recenter_select
        OUT=ROOT/'artifacts/core_port_recenter_20261002';target=OUT/args.mode
        policy_file=Path(__file__).with_name('core_port_recenter_policy.py')
    elif args.strategy=='recheck':
        from core_port_recheck_policy import POLICY as recheck_policy,proposals as recheck_proposals,windows as recheck_windows,select_zoom as recheck_select
        POLICY,proposals,windows,select_zoom=recheck_policy,recheck_proposals,recheck_windows,recheck_select
        OUT=ROOT/'artifacts/core_port_recheck_20261002';target=OUT/args.mode
        policy_file=Path(__file__).with_name('core_port_recheck_policy.py')
    if target.exists():raise FileExistsError('Fresh output required')
    modes=['strict','standard']
    if args.mode!='train':
        locked=load(OUT/'training_decision.json');assert locked['accepted'], 'Training gate failed'
        modes=[locked['mode']]
        if args.mode=='outer':assert load(OUT/'inner/report.json')['qualifies'][modes[0]],'Inner gate failed'
    (target/'config/Ultralytics').mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(target/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
    import torch
    from core_port_resolution_ab_20261002 import DATA,WEIGHT
    from inspection_agent.optional_port_crop_review import CONFIG,read_image,sha
    from inspection_agent.port_tiling import near_artificial_edge
    os.environ['YOLO_CONFIG_DIR']=str(target/'config')
    source=ROOT/('artifacts/core_port_supplement_20261002/train' if args.mode=='train' else 'artifacts/core_precision_full_'+args.mode+'_20261002')
    if args.mode=='train':
        names=load(ROOT/'artifacts/core_port_supplement_20261002/protocol.json')['training_controls']
        if (OUT/'protocol.json').exists():raise FileExistsError('Protocol must not be overwritten')
        save(OUT/'protocol.json',dict(policy=POLICY,training_controls=names,weight_sha256=sha(WEIGHT),
             policy_sha256=sha(policy_file),shared_selector_sha256=sha(dependency_file),strategy=args.strategy,
             context_helper_sha256=sha(Path(__file__).with_name('core_port_context_policy.py')),
             variants_frozen_before_training_evaluation=modes,choice='strict if passes else standard if passes else reject',
             requires_more_tp_no_extra_unmatched_or_normal_cues=True,annotation_crop_selection=False,
             no_val_test_label_parameter_selection=True,primary_and_existing_supplementary_preserved=True,
             maximum_total_cues=10,existing_config=copy.deepcopy(CONFIG)))
    else:names=[r['image'] for r in load(source/'report.json')['cases']]
    protocol=load(OUT/'protocol.json');assert sha(policy_file)==protocol['policy_sha256']
    assert sha(dependency_file)==protocol.get('shared_selector_sha256',protocol['policy_sha256'])
    if 'context_helper_sha256' in protocol:
        assert sha(Path(__file__).with_name('core_port_context_policy.py'))==protocol['context_helper_sha256']
    assert sha(WEIGHT)==protocol['weight_sha256']
    assert CONFIG==protocol['existing_config']
    # Freeze all image-only proposals and verify sources BEFORE reading any labels.
    split='val01' if args.mode=='outer' else 'train01';cases=[]
    for name in names:
        case=load(source/(Path(name).stem+'_predictions.json'))
        path=DATA/'images'/split/name
        assert sha(path)==case['source_sha256'] and case['weight_sha256']==protocol['weight_sha256']
        case['zoom_evidence']=[dict(proposal=p,windows=windows(p,case['predictions']['source_shape'])) for p in proposals(case['predictions'])]
        cases.append(case)
    save(target/'frozen_proposals.json',[dict(image=c['image'],source_sha256=c['source_sha256'],proposals=c['zoom_evidence']) for c in cases])
    if args.mode=='train' and args.strategy=='recheck':
        previous=ROOT/'artifacts/core_port_recenter_20261002/train'
        for case in cases:
            saved=load(previous/(Path(case['image']).stem+'_zoom_predictions.json'))
            assert saved['source_sha256']==case['source_sha256'] and saved['weight_sha256']==case['weight_sha256']
            for entry in case['zoom_evidence']:
                found=[e for e in saved['zoom_evidence'] if e['proposal']==entry['proposal'] and e['windows']==entry['windows']]
                assert len(found)==1
                entry['views']=copy.deepcopy(found[0]['views'])
            case.update(zoom_seconds=0.,new_inference=False,verified_cache_replay=True)
            save(target/(Path(case['image']).stem+'_zoom_predictions.json'),case)
        report=evaluate(cases,split,target,modes)
        report.update(fresh_crop_inference=False,verified_training_cache_replay=True)
        save(target/'report.json',report)
        accepted=[m for m in modes if report['qualifies'][m]]
        save(OUT/'training_decision.json',dict(accepted=bool(accepted),mode=accepted[0] if accepted else None,
              policy_sha256=protocol['policy_sha256'],no_heldout_parameter_selection=True))
        return
    from ultralytics import YOLO
    torch.set_num_threads(4);torch.manual_seed(20260929);model=YOLO(str(WEIGHT))
    assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    start=time.monotonic()
    for index,case in enumerate(cases,1):
        tick=time.monotonic();image=read_image(DATA/'images'/split/case['image']);height,width=image.shape[:2]
        budget_full=len(supplement(case['predictions'])['supplementary'])>=POLICY['maximum_total_supplementary']
        for entry in case['zoom_evidence']:
            if budget_full:
                entry.update(views=[[],[]],inference_skipped='existing_supplementary_budget_full')
                continue
            crops=[image[y:b,x:r] for x,y,r,b in entry['windows']]
            outputs=model.predict(crops,imgsz=POLICY['prediction_imgsz'],conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False)
            # Ultralytics resets CPU threads during its first predictor setup.
            # Restore the intended four-thread cap for subsequent crop calls.
            torch.set_num_threads(4)
            if len(outputs)!=2:raise ValueError('zoom_batch_mismatch')
            views=[]
            for view_id,(output,window) in enumerate(zip(outputs,entry['windows'])):
                rows=[];x,y,_,_=window
                for box in output.boxes:
                    local=list(map(float,box.xyxy[0].tolist()))
                    if near_artificial_edge(local,window,width,height,16):continue
                    l,t,r,b=local
                    rows.append(dict(box_xyxy=[l+x,t+y,r+x,b+y],confidence=float(box.conf.item()),class_id=int(box.cls.item()),zoom_view=view_id))
                views.append(rows)
            entry['views']=views
        case['zoom_seconds']=round(time.monotonic()-tick,2)
        assert sha(DATA/'images'/split/case['image'])==case['source_sha256']
        save(target/(Path(case['image']).stem+'_zoom_predictions.json'),case)
        save(target/'progress.json',dict(completed=index,total=len(cases),seconds=round(time.monotonic()-start,2)))
        print(f"{index}/{len(cases)} {case['image']} proposals={len(case['zoom_evidence'])} seconds={case['zoom_seconds']}",flush=True)
    assert CONFIG==protocol['existing_config'] and sha(WEIGHT)==protocol['weight_sha256']
    report=evaluate(cases,split,target,modes)
    report['seconds']=round(time.monotonic()-start,2);save(target/'report.json',report)
    if args.mode=='train':
        accepted=[m for m in modes if report['qualifies'][m]]
        save(OUT/'training_decision.json',dict(accepted=bool(accepted),mode=accepted[0] if accepted else None,
                 policy_sha256=protocol['policy_sha256'],no_heldout_parameter_selection=True))

if __name__=='__main__':main()
