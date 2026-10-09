"""Whole TRAIN miss cohort: absent geometry vs wrong class. GT is diagnosis only."""
import sys
from pathlib import Path
from collections import Counter
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,DATA,load,save,sha,read_image
from audit_allport480_source import iou


def classify_coverage(target,views,roles):
    per_role={role:{'same':0.,'other':0.,'best_any_row':None} for role in roles}
    reverse={digest:role for role,digest in roles.items()}
    for view in views:
        role=reverse[view['weight_sha256']];best=per_role[role]
        for row in view['predictions']['merged_predictions']:
            if row['confidence']<=.05:continue
            value=iou(row['box_xyxy'],target['box']);kind='same' if row['class_id']==target['class_id'] else 'other'
            before=max(best['same'],best['other']);best[kind]=max(best[kind],value)
            if value>before:best['best_any_row']=dict(box_xyxy=row['box_xyxy'],class_id=row['class_id'],confidence=row['confidence'],iou=value)
    same=max((v['same'] for v in per_role.values()),default=0.)
    other=max((v['other'] for v in per_role.values()),default=0.)
    reason=('some_correct_class_geometry_but_missing_other_voters' if same>=.5 else
            'localized_port_only_wrong_detector_class' if other>=.5 else
            'nearby_raw_geometry_below_class_IoU50' if max(same,other)>=.3 else
            'no_any_class_raw_geometry_at_IoU30')
    return dict(reason=reason,per_actual_role=per_role,distinct_roles_with_same_IoU50=sum(v['same']>=.5 for v in per_role.values()),
                distinct_roles_with_other_IoU50=sum(v['other']>=.5 for v in per_role.values()))


def main():
    from PIL import Image,ImageDraw,ImageFont
    import cv2
    source=ROOT/'artifacts/allport480_teacher_source_20261005';auditpath=ROOT/'artifacts/allport480_teacher_source_audit_20261005/report.json'
    funnelpath=ROOT/'artifacts/allport480_teacher_miss_funnel_20261005/report.json';out=ROOT/'artifacts/allport480_missing_class_coverage_20261005'
    if out.exists():raise FileExistsError('preserve full missing-class diagnosis')
    audit=load(auditpath);report=load(source/'report.json');funnel=load(funnelpath);roles=load(source/'protocol.json')['roles']
    if audit['status']!='pass' or audit['source_report_sha256']!=sha(source/'report.json') or funnel['source_report_sha256']!=sha(source/'report.json'):raise ValueError('requires completed independent TRAIN trial')
    pins={str(p):sha(p) for p in [Path(__file__),source/'report.json',source/'protocol.json',auditpath,funnelpath]};rows=[];counts=Counter();out.mkdir()
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for item in funnel['remaining_misses']:
        if item['reason']=='policy_shared_budget_full':continue
        name=item['image'];path=source/'train'/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path)
        if pins[str(path)]!=audit['source_case_sha256'][str(path)]:raise ValueError('audited detector outputs changed')
        case=load(path)
        if not case['eligible']:raise ValueError('unobserved views cannot be called absent detections')
        detail=classify_coverage(item,case['new_voter_views'],roles);counts[detail['reason']]+=1
        imagepath=DATA/'images/train01'/name;pins[str(imagepath)]=sha(imagepath)
        if pins[str(imagepath)]!=case['source_sha256']:raise ValueError('source image drift')
        pixels=read_image(imagepath);h,w=pixels.shape[:2];box=item['box'];side=384
        x=max(0,min(w-side,round((box[0]+box[2])/2-side/2)));y=max(0,min(h-side,round((box[1]+box[3])/2-side/2)))
        canvas=Image.new('RGB',(1182,470),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'{name} GT{item["target_index"]} class{item["class_id"]}: '+detail['reason'],fill='black',font=font)
        for index,role in enumerate(roles):
            crop=pixels[y:y+side,x:x+side].copy()
            def line(coords,color,label):
                l,t,r,b=map(round,coords);cv2.rectangle(crop,(l-x,t-y),(r-x,b-y),color,2)
                if x<=l<x+side and y<=t<y+side:cv2.putText(crop,label,(l-x,max(15,t-y-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            line(box,(170,170,170),'GT')
            best=detail['per_actual_role'][role];nearest=best['best_any_row']
            if nearest and nearest['iou']>0:line(nearest['box_xyxy'],(220,105,0) if nearest['class_id']==item['class_id'] else (30,30,220),f'c{nearest["class_id"]} IoU{nearest["iou"]:.2f}')
            draw.text((index*394+8,31),f'{role} same{best["same"]:.2f} other{best["other"]:.2f}',fill='black',font=font)
            canvas.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(index*394+5,60))
        card=out/f'{Path(name).stem}_GT{item["target_index"]}.png';canvas.save(card)
        rows.append(dict(image=name,target_index=item['target_index'],class_id=item['class_id'],GT_box=box,card=str(card),**detail))
    if len(rows)!=14 or any(sha(p)!=d for p,d in pins.items()):raise ValueError('full eligible source miss cohort changed')
    save(out/'report.json',dict(status='complete_post_selection_diagnosis',eligible_missing_targets=len(rows),taxonomy=dict(counts),cases=rows,pins=pins,
        same_weight_views_one_support=True,no_model_inference=True,no_selection_or_GT_changes=True,no_validation_inputs=True,
        manual_review='pending',no_deployment=True,field_accuracy=None,
        warning='All14 eligible TRAIN misses, not field accuracy or a proposal-class relabeling rule. Raw geometry needs separate class and vote checks.'))
    print(dict(status='complete_post_selection_diagnosis',eligible_misses=len(rows),taxonomy=dict(counts)))


if __name__=='__main__':main()
