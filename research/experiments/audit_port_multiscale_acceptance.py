"""Independent post-inference scoring and gain/loss cards; no policy changes."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha,read_image
BASE=ROOT/'artifacts/port_training_multiscale_20261002';OLD=ROOT/'artifacts/core_port_recheck_20261002'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.
def matches(predictions,targets):
    pairs=sorted(((overlap(p['box_xyxy'],t['box']),pi,ti) for pi,p in enumerate(predictions)
        for ti,t in enumerate(targets) if p['class_id']==t['class_id']),reverse=True)
    usedp=set();usedt=set();assigned=[]
    for value,pi,ti in pairs:
        if value>=.5 and pi not in usedp and ti not in usedt:
            usedp.add(pi);usedt.add(ti);assigned.append(dict(prediction=pi,target=ti,iou=value))
    return usedt,set(range(len(predictions)))-usedp,assigned
def metric(predictions,targets):
    usedt,unmatched,_=matches(predictions,targets)
    return dict(tp=len(usedt),unmatched=len(unmatched),fn=len(targets)-len(usedt),predictions=len(predictions),targets=len(targets))

def render_card(source,targets,old,new,focus,title,path):
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    image=read_image(source);h,w=image.shape[:2];l,t,r,b=focus
    side=max(384,round(max(r-l,b-t)+320));cx=(l+r)/2;cy=(t+b)/2
    x=max(0,min(w-side,round(cx-side/2)));y=max(0,min(h-side,round(cy-side/2)))
    right=min(w,x+side);bottom=min(h,y+side)
    panels=[]
    for selected in (old,new):
        canvas=image[y:bottom,x:right].copy()
        def draw(box,color,text):
            L,T,R,B=map(round,box);cv2.rectangle(canvas,(L-x,T-y),(R-x,B-y),color,2)
            if x<=L<right and y<=T<bottom:cv2.putText(canvas,text,(L-x,max(17,T-y-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
        for target in targets:draw(target['box'],(190,190,190),'GT c'+str(target['class_id']))
        for row in selected:draw(row['box_xyxy'],(255,150,0),'c%d %.3f'%(row['class_id'],row['confidence']))
        panel=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));panel.thumbnail((480,480))
        panels.append(panel)
    card=Image.new('RGB',(980,550),'white');draw=ImageDraw.Draw(card);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    draw.text((10,5),title,fill='black',font=font)
    for offset,panel,label in ((10,panels[0],'Accepted teacher'),(500,panels[1],'Fixed last.pt candidate')):
        draw.text((offset,30),label,fill='black',font=font);card.paste(panel,(offset,60))
    card.save(path,quality=94)

def main():
    output=BASE/'acceptance_audit_20261003'
    if output.exists():raise FileExistsError('Preserve existing audits')
    output.mkdir();full=load(BASE/'full/report.json');weight=Path(full['weights']['last.pt']['path'])
    assert sha(weight)==full['weights']['last.pt']['sha256']
    report=dict(status='complete',candidate_sha256=sha(weight),splits={},cards=[],
        independent_geometry_scoring=True,operator_confirmation=False,production_approved=False)
    for split in ('inner','outer'):
        folder=BASE/'evaluation'/split;result=load(folder/'report.json');protocol=load(folder/'protocol.json')
        assert result['status']=='complete' and not result['baseline_replay_only']
        assert {p:sha(Path(p)) for p in protocol['pins']}==protocol['pins']
        old_report=load(OLD/split/'report.json');source_split='train01' if split=='inner' else 'val01'
        totals={k:0 for k in ('tp','unmatched','fn','predictions','targets')};old_totals=dict(totals);rows=[]
        stage_totals={version:{stage:dict(totals) for stage in ('primary','primary_plus_consensus','all_including_recheck')}
            for version in ('baseline','candidate')}
        card_requests={'gained':[],'lost':[],'unmatched':[]};normal_cues=0
        for entry in result['cases']:
            name=entry['image'];stem=Path(name).stem
            raw=load(folder/(stem+'_zoom_predictions.json'));new_saved=load(folder/(stem+'_evaluation.json'))
            old_raw=load(OLD/split/(stem+'_zoom_predictions.json'));old_saved=load(OLD/split/(stem+'_evaluation.json'))
            assert raw['source_sha256']==old_raw['source_sha256'] and raw['weight_sha256']==report['candidate_sha256']
            h,w=raw['predictions']['source_shape'];assert [h,w]==old_raw['predictions']['source_shape']
            label=DATA/'labels'/source_split/(stem+'.txt');assert sha(label)==new_saved['label_sha256']==old_saved['label_sha256']
            targets=[]
            for line in label.read_text(encoding='utf-8').splitlines():
                cls,cx,cy,bw,bh=map(float,line.split())
                if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
            old=old_saved['selected']['strict']['all_predictions'];new=new_saved['selected']['all_predictions']
            for version,selection in (('baseline',old_saved['selected']['strict']),('candidate',new_saved['selected'])):
                for stage,stage_rows in (('primary',selection['primary']),
                    ('primary_plus_consensus',selection['primary']+selection['supplementary']),
                    ('all_including_recheck',selection['all_predictions'])):
                    stage_metric=metric(stage_rows,targets)
                    for key in totals:stage_totals[version][stage][key]+=stage_metric[key]
            assert len(new_saved['selected']['primary'])<=5 and len(new_saved['selected']['supplementary'])+len(new_saved['selected']['zoom'])<=5
            current=metric(new,targets);baseline=metric(old,targets)
            assert current==entry['metrics']==new_saved['metrics']
            assert baseline==old_saved['metrics']['strict']
            for key in totals:totals[key]+=current[key];old_totals[key]+=baseline[key]
            if name.startswith('normal_'):normal_cues+=len(new)
            old_targets,old_unmatched,_=matches(old,targets);new_targets,new_unmatched,pairs=matches(new,targets)
            gained=sorted(new_targets-old_targets);lost=sorted(old_targets-new_targets)
            rows.append(dict(image=name,baseline=baseline,candidate=current,gained_target_indices=gained,
                lost_target_indices=lost,new_unmatched_prediction_indices=sorted(new_unmatched),matching=pairs))
            for kind,indices in (('gained',gained),('lost',lost),('unmatched',sorted(new_unmatched))):
                if indices and len(card_requests[kind])<2:
                    focus=new[indices[0]]['box_xyxy'] if kind=='unmatched' else targets[indices[0]]['box']
                    card_requests[kind].append((name,targets,old,new,focus))
        assert totals==result['summary'] and old_totals==result['accepted_baseline']==old_report['summary']['strict']
        report['splits'][split]=dict(images=len(rows),baseline=old_totals,candidate=totals,normal_cues=normal_cues,
            stage_totals=stage_totals,
            gained_targets=sum(len(r['gained_target_indices']) for r in rows),lost_targets=sum(len(r['lost_target_indices']) for r in rows),
            acceptance=result['acceptance'],cases=rows)
        for kind,requests in card_requests.items():
            for index,(name,targets,old,new,focus) in enumerate(requests,1):
                path=output/f'{split}_{kind}_{index}_{Path(name).stem}.jpg'
                render_card(DATA/'images'/source_split/name,targets,old,new,focus,f'{split} {name}: {kind}',path)
                report['cards'].append(dict(path=str(path),split=split,reason=kind,image=name))
    report['source_only_acceptance_passed']=all(v['acceptance']['source_only_pass'] for v in report['splits'].values())
    report['scope']='Previously inspected same-camera port localization, not fault/continuity/cross-cabinet accuracy'
    save(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('splits','cards')},indent=2),flush=True)

if __name__=='__main__':main()
