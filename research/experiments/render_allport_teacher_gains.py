"""Visualize all gains in one locked collection snapshot; NOT acceptance.

Never mutate live inference code, inputs or outputs. Inspect original source
pixels and original GT, showing accepted/mainline vs research vs new candidate.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,DATA,load,save,sha,read_image
from audit_allport480_source import score,iou
from current_port_baseline_audit import BASE


def main(source,out):
    from PIL import Image,ImageDraw,ImageFont
    import cv2
    source=source.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('preserve previous visual snapshot')
    snapshotpath=source/('report.json' if (source/'report.json').exists() else 'partial_metrics.json')
    raw=snapshotpath.read_bytes();snapshot=json.loads(raw.decode('utf-8'));snapshot_sha=hashlib.sha256(raw).hexdigest()
    entries={r['image']:r for r in load(BASE/'train/report.json')['cases']}
    records=[];pins={str(Path(__file__)):sha(Path(__file__))};out.mkdir()
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for item in snapshot['cases']:
        if not item['gained']:continue
        name=item['image'];casepath=source/'train'/(Path(name).stem+'_predictions.json');pins[str(casepath)]=sha(casepath);case=load(casepath)
        imagepath=DATA/'images/train01'/name;labelpath=DATA/'labels/train01'/(Path(name).stem+'.txt')
        pins[str(imagepath)]=sha(imagepath);pins[str(labelpath)]=sha(labelpath)
        if pins[str(imagepath)]!=case['source_sha256'] or pins[str(labelpath)]!=entries[name]['label_sha256']:raise ValueError('source/GT drift')
        pixels=read_image(imagepath);h,w=pixels.shape[:2];targets=[]
        for line in labelpath.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        _,current_hits=score(case['current']['all_predictions'],targets);trial_metric,trial_hits=score(case['trial']['all_predictions'],targets)
        if sorted(trial_hits-current_hits)!=item['gained'] or trial_metric!=item['trial']:raise ValueError('snapshot gain replay differs')
        for ti in item['gained']:
            target=targets[ti];box=target['box'];side=384
            left=max(0,min(w-side,round((box[0]+box[2])/2-side/2)));top=max(0,min(h-side,round((box[1]+box[3])/2-side/2)))
            card=Image.new('RGB',(1182,468),'white');draw=ImageDraw.Draw(card)
            draw.text((10,6),f'{name} GT{ti} class{target["class_id"]}: partial-source gain, not final acceptance',fill='black',font=font)
            for column,(version,title) in enumerate([('original','Accepted295 prefix'),('current','Research298 prefix'),('trial','Teacher-replacement candidate')]):
                crop=pixels[top:top+side,left:left+side].copy()
                def boxline(coords,color,text):
                    l,t,r,b=map(round,coords);cv2.rectangle(crop,(l-left,t-top),(r-left,b-top),color,2)
                    if left<=l<left+side and top<=t<top+side:cv2.putText(crop,text,(l-left,max(16,t-top-4)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
                boxline(box,(170,170,170),'GT')
                for row in case[version]['all_predictions']:
                    if iou(row['box_xyxy'],[left,top,left+side,top+side])>0:
                        boxline(row['box_xyxy'],(220,105,0),f'c{row["class_id"]}')
                draw.text((column*394+8,32),title,fill='black',font=font)
                card.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(column*394+5,60))
            filename=f'{Path(name).stem}_GT{ti}.png';card.save(out/filename)
            # Actual GT-match partners, not probability alone.
            matches=[dict(row_index=pi,box_xyxy=row['box_xyxy'],iou=iou(row['box_xyxy'],box))
                     for pi,row in enumerate(case['trial']['all_predictions']) if row['class_id']==target['class_id'] and iou(row['box_xyxy'],box)>=.5]
            records.append(dict(image=name,target_index=ti,class_id=target['class_id'],GT_box=box,crop_native_xyxy=[left,top,left+side,top+side],card=str(out/filename),candidate_partners=matches))
    if any(sha(p)!=d for p,d in pins.items()):raise ValueError('immutable gain review inputs changed')
    save(out/'report.json',dict(status='complete_visual_snapshot_not_acceptance',snapshot_path=str(snapshotpath),snapshot_bytes_sha256=snapshot_sha,
        collected_sources=len(snapshot['cases']),total_sources=192,records=records,pins=pins,new_detector_inference=False,
        source_trial_complete='summary' in snapshot,no_selection_changes=True,no_deployment=True,field_accuracy=None,
        warning='Cards show original GT localization gains; manual inspection and final source/holdout gates remain separate.'))
    print(dict(status='complete_visual_snapshot_not_acceptance',collected_sources=len(snapshot['cases']),cards=len(records)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'artifacts/allport480_teacher_source_20261005')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/allport480_teacher_gain_review_20261005');args=parser.parse_args();main(args.source,args.output)
