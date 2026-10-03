"""One-variable core detector experiment on original image-group inner holdout."""
import os,sys,json,copy,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/core_port_resolution_ab_20261002'
sys.path.insert(0,str(REPO))
os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
import torch
from inspection_agent.optional_port_crop_review import CONFIG,sha,read_image,predict
from inspection_agent.port_tiling import box_iou
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
WEIGHT=REPO/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
MANIFEST=REPO/'data/derived/port_crop_training_20260929/dataset_manifest.json'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def ranking(rows,budget):
    chosen=[p for p in rows if p['confidence']>.25]
    chosen.sort(key=lambda p:(-p['confidence'],(p['box_xyxy'][2]-p['box_xyxy'][0])*(p['box_xyxy'][3]-p['box_xyxy'][1]),*p['box_xyxy'],p['class_id']))
    return chosen if budget is None else chosen[:budget]
def score(predictions,targets):
    possibilities=sorted([(box_iou(p['box_xyxy'],t['box']),pi,ti) for pi,p in enumerate(predictions)
        for ti,t in enumerate(targets) if p['class_id']==t['class_id']],reverse=True)
    usedp=set();usedt=set()
    for iou,pi,ti in possibilities:
        if iou>=.5 and pi not in usedp and ti not in usedt:usedp.add(pi);usedt.add(ti)
    return dict(tp=len(usedp),unmatched=len(predictions)-len(usedp),fn=len(targets)-len(usedt),
                predictions=len(predictions),targets=len(targets))
def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    (OUT/'config/Ultralytics').mkdir(parents=True)
    manifest=load(MANIFEST);train={r['source_image'] for r in manifest['records'] if r['split']=='train'}
    val={r['source_image'] for r in manifest['records'] if r['split']=='val'}
    assert train.isdisjoint(val)
    names=sorted(n for n in val if n.startswith('disconnected_'))
    for kind in ('normal','damaged','misrouted'):names+=sorted(n for n in val if n.startswith(kind+'_'))[:2]
    paths=[DATA/'images/train01'/n for n in names]
    frozen=load(REPO/'config/port_crop_calibration_frozen_20260930.json')
    assert sha(WEIGHT)==frozen['fingerprints']['weight']
    hashes={str(p):sha(p) for p in paths+[WEIGHT,MANIFEST,REPO/'inspection_agent/optional_port_crop_review.py']}
    original=copy.deepcopy(CONFIG)
    save(OUT/'protocol.json',dict(images=names,source_split='train01_original_image_group_inner_val',
        weight_sha256=sha(WEIGHT),baseline_config=original,experimental_predict_imgsz=1280,
        threshold=.25,budgets=[1,5,'all'],selection='all heldout disconnected + first two each normal/damaged/misrouted',
        no_test_or_outer_val_access=True,labels_used_for_inference=False,training=False,
        deployment_requires_gain_without_more_unmatched_at_same_budget=True,hashes=hashes))
    from ultralytics import YOLO
    torch.set_num_threads(4);torch.manual_seed(20260929)
    model=YOLO(str(WEIGHT));assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    cases=[];start=time.monotonic()
    try:
        for path in paths:
            case=dict(image=path.name,predictions={},seconds={})
            image=read_image(path)
            for size in (960,1280):
                CONFIG['predict_imgsz']=size;tick=time.monotonic()
                case['predictions'][str(size)]=predict(model,image)
                case['seconds'][str(size)]=round(time.monotonic()-tick,2)
            save(OUT/(path.stem+'_predictions.json'),case);cases.append(case)
            save(OUT/'progress.json',dict(completed=len(cases),total=len(paths),seconds=round(time.monotonic()-start,2)))
            print(f"{len(cases)}/{len(paths)} {path.name} timings={case['seconds']}",flush=True)
    finally:
        CONFIG.clear();CONFIG.update(original)
    # All labels opened only after all inference completes, not for parameter selection.
    aggregates={str(size):{str(budget):dict(tp=0,unmatched=0,fn=0,predictions=0,targets=0) for budget in (1,5,None)} for size in (960,1280)}
    for case in cases:
        targets=[];label=DATA/'labels/train01'/(Path(case['image']).stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736]))
        case['evaluation']={}
        for size in (960,1280):
            stage={}
            for budget in (1,5,None):
                metrics=score(ranking(case['predictions'][str(size)]['merged_predictions'],budget),targets)
                stage[str(budget)]=metrics
                for key,value in metrics.items():aggregates[str(size)][str(budget)][key]+=value
            case['evaluation'][str(size)]=stage
        case['label_sha256']=sha(label)
        save(OUT/(Path(case['image']).stem+'_evaluation.json'),case['evaluation'])
    assert {p:sha(p) for p in hashes}==hashes and CONFIG==original
    outcome={}
    for budget in ('1','5','None'):
        a=aggregates['960'][budget];b=aggregates['1280'][budget]
        outcome[budget]=dict(tp_gain=b['tp']-a['tp'],unmatched_change=b['unmatched']-a['unmatched'],
            qualifies_for_further_validation=b['tp']>a['tp'] and b['unmatched']<=a['unmatched'])
    save(OUT/'report.json',dict(status='complete',aggregate=aggregates,outcome=outcome,
        cases=[dict(image=c['image'],seconds=c['seconds'],evaluation=c['evaluation']) for c in cases],
        seconds=round(time.monotonic()-start,2),formal_policy_changed=False,field_accuracy_claimed=False,
        caution='Inner validation already used for original model selection, not independent unseen evaluation'))
    print(json.dumps(dict(aggregate=aggregates,outcome=outcome),indent=2),flush=True)
if __name__=='__main__':main()
