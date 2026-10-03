"""Fixed source-only localization comparison; never changes production calibration."""
import argparse,copy,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
BASE=ROOT/'artifacts/port_training_multiscale_20261002'
OLD=ROOT/'artifacts/core_port_recheck_20261002'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import CONFIG,sha,read_image,predict
from inspection_agent.port_tiling import near_artificial_edge
from core_port_recheck_policy import POLICY,proposals,windows,select_zoom
from core_port_supplement_policy import supplement

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(p)

def acceptance(mode,totals,normal,protocol):
    if mode=='train':return dict(diagnostic_only=True,source_only_pass=False)
    rule=protocol[mode+'_acceptance']
    valid_targets=totals['targets']==rule['targets']
    improves=totals['tp']>rule['require_tp_strictly_greater'] if mode=='inner' else totals['tp']>=rule['minimum_tp']
    checks=dict(target_count=valid_targets,hit_count=improves,
        unmatched=totals['unmatched']<=rule['maximum_unmatched'],normal_cues=normal<=rule['maximum_normal_cues'])
    return dict(checks=checks,source_only_pass=all(checks.values()),production_approved=False)

def aggregate(cases,split,target):
    # Reuse precisely the old one-to-one scoring, only AFTER image-only predictions.
    from core_port_resolution_ab_20261002 import score
    totals={k:0 for k in ('tp','unmatched','fn','predictions','targets')};groups={};rows=[]
    for case in cases:
        name=case['image'];h,w=case['predictions']['source_shape'];label=DATA/'labels'/split/(Path(name).stem+'.txt')
        targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        selected=select_zoom(case,'strict')
        assert len(selected['primary'])<=5 and len(selected['supplementary'])+len(selected['zoom'])<=5
        metric=score(selected['all_predictions'],targets);kind=name.split('_')[0]
        group=groups.setdefault(kind,{k:0 for k in totals})
        for k in totals:totals[k]+=metric[k];group[k]+=metric[k]
        result=dict(image=name,selected=selected,metrics=metric,label_sha256=sha(label))
        if target is not None:save(target/(Path(name).stem+'_evaluation.json'),result)
        rows.append(dict(image=name,metrics=metric))
    return dict(summary=totals,groups=groups,cases=rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','inner','outer'],required=True)
    parser.add_argument('--verify-baseline',action='store_true');args=parser.parse_args()
    mode=args.mode;protocol=load(BASE/'evaluation_protocol.json');data_protocol=load(BASE/'protocol.json')
    assert protocol['freeze_before_training'] and protocol['checkpoint']=='last.pt'
    old_protocol=load(OLD/'protocol.json');assert CONFIG==old_protocol['existing_config']
    assert POLICY==old_protocol['policy']
    train=set(data_protocol['train_sources']);inner=set(data_protocol['inner_val_sources']);assert train.isdisjoint(inner)
    expected_names=(old_protocol['training_controls'] if mode=='train' else
        data_protocol['inner_val_sources'] if mode=='inner' else [r['image'] for r in load(OLD/'outer/report.json')['cases']])
    assert len(expected_names)=={'train':32,'inner':48,'outer':30}[mode]
    if mode=='train':assert set(expected_names)<=train
    if mode=='inner':assert set(expected_names)==inner
    split='val01' if mode=='outer' else 'train01'
    pins={str(p):sha(p) for p in [Path(__file__),BASE/'evaluation_protocol.json',BASE/'protocol.json',
        REPO/'config/port_crop_calibration_frozen_20260930.json',
        REPO/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt',
        REPO/'inspection_agent/optional_port_crop_review.py',REPO/'inspection_agent/port_tiling.py',
        ROOT/'experiments/core_port_precision_policy.py',ROOT/'experiments/core_port_supplement_policy.py',
        ROOT/'experiments/core_port_zoom_policy.py',ROOT/'experiments/core_port_context_policy.py',
        ROOT/'experiments/core_port_recenter_policy.py',ROOT/'experiments/core_port_recheck_policy.py',
        ROOT/'experiments/core_port_resolution_ab_20261002.py']}
    old_cases=[]
    for name in expected_names:
        case=load(OLD/mode/(Path(name).stem+'_zoom_predictions.json'))
        assert case['image']==name and case['weight_sha256']==old_protocol['weight_sha256']
        source=DATA/'images'/split/name;assert sha(source)==case['source_sha256']
        pins[str(source)]=case['source_sha256'];old_cases.append(case)
    target=BASE/('baseline_replay' if args.verify_baseline else 'evaluation')/mode
    if target.exists():raise FileExistsError('Fresh output required; preserve prior results')
    candidate=None
    if not args.verify_baseline:
        full=load(BASE/'full/report.json')
        assert full['status']=='complete' and full['completed_epochs']==2
        assert full['nonzero_gradient_steps']>0 and full['changed_trainable_parameters']>0
        assert full['evaluation_protocol_sha256']==sha(BASE/'evaluation_protocol.json')
        assert full['initialization_sha256']==old_protocol['weight_sha256']
        candidate=Path(full['weights']['last.pt']['path']);assert sha(candidate)==full['weights']['last.pt']['sha256']
        assert sha(candidate)!=full['initialization_sha256'];pins[str(candidate)]=sha(candidate)
    target.mkdir(parents=True)
    save(target/'protocol.json',dict(mode=mode,images=expected_names,pins=pins,config=copy.deepcopy(CONFIG),policy=copy.deepcopy(POLICY),
        candidate=str(candidate) if candidate else None,checkpoint_selection='fixed last.pt',
        labels_not_used_for_inference=True,source_only=True,reference_gates_not_tested=True,production_approved=False))
    started=time.monotonic();cases=[]
    if args.verify_baseline:
        cases=old_cases
    else:
        (target/'config/Ultralytics').mkdir(parents=True)
        shutil.copy2('C:/Windows/Fonts/arial.ttf',target/'config/Ultralytics/Arial.ttf')
        os.environ.update(YOLO_CONFIG_DIR=str(target/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4);model=YOLO(str(candidate))
        assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
        class CappedModel:
            def predict(self,*a,**kw):
                result=model.predict(*a,**kw);torch.set_num_threads(4);return result
        capped=CappedModel()
        for index,old_case in enumerate(old_cases,1):
            name=old_case['image'];image=read_image(DATA/'images'/split/name);h,w=image.shape[:2];tick=time.monotonic()
            raw=predict(capped,image)
            case=dict(image=name,source_sha256=old_case['source_sha256'],weight_sha256=pins[str(candidate)],
                predictions=raw,zoom_evidence=[],image_only_inference=True)
            full_budget=len(supplement(raw)['supplementary'])>=5
            for seed in proposals(raw):
                views=[];crops=windows(seed,raw['source_shape'])
                if full_budget:views=[[],[]]
                else:
                    outputs=capped.predict([image[y:b,x:r] for x,y,r,b in crops],imgsz=960,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False)
                    assert len(outputs)==2
                    for view_id,(output,window) in enumerate(zip(outputs,crops)):
                        rows=[];x,y,_,_=window
                        for box in output.boxes:
                            local=list(map(float,box.xyxy[0].tolist()))
                            if near_artificial_edge(local,window,w,h,16):continue
                            l,t,r,b=local;rows.append(dict(box_xyxy=[l+x,t+y,r+x,b+y],confidence=float(box.conf.item()),class_id=int(box.cls.item()),zoom_view=view_id))
                        views.append(rows)
                case['zoom_evidence'].append(dict(proposal=seed,windows=crops,views=views))
            case['zoom_seconds']=round(time.monotonic()-tick,2);cases.append(case)
            save(target/(Path(name).stem+'_zoom_predictions.json'),case)
            save(target/'progress.json',dict(status='running',completed=index,total=len(expected_names),seconds=round(time.monotonic()-started,2)))
            print(f'{index}/{len(expected_names)} {name}: {case["zoom_seconds"]}s',flush=True)
    baseline=aggregate(old_cases,split,None)
    old_report=load(OLD/mode/'report.json')
    assert baseline['summary']==old_report['summary']['strict']
    assert baseline['groups']=={k:v['strict'] for k,v in old_report['groups'].items()}
    result=aggregate(cases,split,target)
    normal=result['groups'].get('normal',{}).get('predictions',0)
    result.update(status='complete',mode=mode,seconds=round(time.monotonic()-started,2),
        accepted_baseline=baseline['summary'],acceptance=acceptance(mode,result['summary'],normal,protocol),
        baseline_replay_only=args.verify_baseline,source_only=True,production_approved=False,
        scope=protocol['scope'],warning='Previously inspected same-camera datasets; not unseen-cabinet or physical fault accuracy.')
    assert {p:sha(Path(p)) for p in pins}==pins
    save(target/'report.json',result);save(target/'progress.json',dict(status='complete',completed=len(cases),total=len(cases)))
    print(json.dumps(dict(summary=result['summary'],acceptance=result['acceptance']),indent=2),flush=True)

if __name__=='__main__':main()
