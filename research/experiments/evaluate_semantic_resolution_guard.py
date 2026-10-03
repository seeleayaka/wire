"""Frozen semantic gate ONLY on strict1280 additions, ALL192 training sources."""
import json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path[:0]=[str(REPO),str(REPO/'prototype')]
OUT=ROOT/'artifacts/semantic_resolution_guard_20261003'
HIGH=ROOT/'artifacts/port_high_resolution_support_20261003'
EXTRA=ROOT/'artifacts/remaining_training_ports_20261003'
HEAD=ROOT/'artifacts/port_semantic_verifier_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
os.environ['HF_HUB_OFFLINE']='1'
from inspection_agent.optional_port_crop_review import sha,read_image
from port_semantic_verifier import embeddings
from port_semantic_support_policy import append_semantic
from audit_port_multiscale_acceptance import metric,matches
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    if OUT.exists():raise FileExistsError('Preserve experiment')
    assert load(EXTRA/'report.json')['status']=='complete'
    assert load(HEAD/'report.json')['qualifies_crop_feasibility']
    headprotocol=load(HEAD/'protocol.json');fold_for={name:i%3 for i,name in enumerate(headprotocol['train_sources'])}
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_semantic_verifier.py'),
        Path(__file__).with_name('port_semantic_support_policy.py'),HEAD/'protocol.json',HEAD/'report.json',EXTRA/'report.json')}
    for p,digest in headprotocol['pins'].items():assert sha(Path(p))==digest;pins[p]=digest
    headpaths=[HEAD/f'fold{fold}_head.pt' for fold in range(3)]
    for p in headpaths:pins[str(p)]=sha(p)
    inputs=[]
    for folder in (HIGH/'train',HIGH/'extended',EXTRA):
        for row in load(folder/'report.json')['cases']:
            name=row['image'];path=folder/(Path(name).stem+'_predictions.json');case=load(path)
            pins[str(path)]=sha(path);source=DATA/'images/train01'/name
            assert name in fold_for
            alternative=case['alternative'];assert alternative['source_sha256']==pins[str(source)]
            inputs.append(dict(image=name,source=source,current=case['current'],candidates=case['trial']['resolution_additions'],
                source_shape=alternative['predictions']['source_shape'],label_sha256=row['label_sha256']))
    assert len(inputs)==192 and set(r['image'] for r in inputs)==set(fold_for)
    OUT.mkdir();save(OUT/'protocol.json',dict(pins=pins,images=[r['image'] for r in inputs],
        preserve_three_model_baseline=True,only_strict1280_additions=True,no_new_low_score_candidates=True,
        source_group_oof_semantic_threshold=.98,all192_training_sources=True,net_tp_gain_no_unmatched_increase=True,
        no_old_target_loss=True,normal_zero=True,inner_outer_pending=True,reference_pending=True,
        rejected_raw_resolution_trial_not_deployed=True,field_accuracy=False,production_changed=False))
    import torch
    import dino_feature_diff as dino
    torch.set_num_threads(2);model=dino._model();model.requires_grad_(False);torch.set_num_threads(2)
    heads=[]
    for path in headpaths:
        head=torch.nn.Linear(1536,3);head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));head.eval();heads.append(head)
    records=[];started=time.monotonic()
    for index,row in enumerate(inputs):
        candidates=row['candidates'];name=row['image'];current=row['current']
        if candidates:
            features=embeddings(model,read_image(row['source']),[r['box_xyxy'] for r in candidates])
            with torch.inference_mode():probabilities=heads[fold_for[name]](features).softmax(dim=1).tolist()
        else:probabilities=[]
        trial=append_semantic(current,candidates,probabilities)
        save(OUT/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,candidates=candidates,probabilities=probabilities))
        h,w=row['source_shape'];targets=[];label=DATA/'labels/train01'/(Path(name).stem+'.txt')
        assert sha(label)==row['label_sha256']
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        old,new=current['all_predictions'],trial['all_predictions'];oldhits,newhits=matches(old,targets)[0],matches(new,targets)[0]
        record=dict(image=name,metrics=dict(current=metric(old,targets),trial=metric(new,targets)),
            gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits),additions=len(trial['semantic_additions']),probabilities=probabilities)
        records.append(record)
        save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=index+1,total=192))
    assert {p:sha(Path(p)) for p in pins}==pins
    totals={s:{key:sum(r['metrics'][s][key] for r in records) for key in ('tp','unmatched','fn','predictions','targets')} for s in ('current','trial')}
    normal=sum(r['metrics']['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
    passed=totals['trial']['tp']>totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
    save(OUT/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,
        seconds=round(time.monotonic()-started,2),reference_pending=True,validation_pending=True,field_accuracy=False,production_changed=False))
    save(OUT/'progress.json',dict(status='complete',qualifies=passed,summary=totals))
if __name__=='__main__':main()
