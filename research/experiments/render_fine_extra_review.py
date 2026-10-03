"""All nine rejected fine additions: paired diagnostic cards, no label changes."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,DATA,load,save,sha,read_image,REFERENCE_SHA
from inspection_agent.paired_port_features import expected_in_source
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import overlap
OUT=ROOT/'artifacts/fine_extra_visual_review_20261004'
SOURCE=ROOT/'artifacts/fine_native_consensus_20261004/train'


def main():
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    cv2.setNumThreads(1)
    if OUT.exists():raise FileExistsError('Preserve diagnostic cards')
    report=load(SOURCE/'report.json');assert report['qualifies'] is False
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    reference=read_image(referencepath);pins={str(referencepath):REFERENCE_SHA,str(SOURCE/'report.json'):sha(SOURCE/'report.json'),str(Path(__file__)):sha(Path(__file__))}
    OUT.mkdir();entries={r['image']:r for r in load(BASE/'train/report.json')['cases']};rows=[];cards=[]
    fontpath=Path('C:/Windows/Fonts/arial.ttf');font=ImageFont.truetype(str(fontpath),19)
    for case in report['cases']:
        path=SOURCE/(Path(case['image']).stem+'_predictions.json');data=load(path)
        additions=data['trial']['paired_semantic_additions']
        if not additions:continue
        pins[str(path)]=sha(path);imagepath=DATA/'images/train01'/case['image'];pins[str(imagepath)]=sha(imagepath);image=read_image(imagepath)
        expected,mask=expected_in_source(reference,data['alignment']['source_to_reference_homography'],image.shape[:2])
        targets=read_targets('train',case['image'],list(image.shape[:2]),entries[case['image']]['label_sha256'],pins)
        for index,p in enumerate(additions):
            l,t,r,b=p['box_xyxy'];cx,cy=(l+r)/2,(t+b)/2;side=max(64,max(r-l,b-t)*3)
            left,top=int(np.floor(cx-side/2)),int(np.floor(cy-side/2));right,bottom=int(np.ceil(cx+side/2)),int(np.ceil(cy+side/2))
            hit=max((overlap(p['box_xyxy'],q['box']) for q in targets if q['class_id']==p['class_id']),default=0.)
            card=Image.new('RGB',(1200,410),'white');draw=ImageDraw.Draw(card)
            draw.text((12,8),f"{case['image']} | class {p['class_id']} | p={p['confidence']:.4f} | weak-label best IoU={hit:.3f}",fill='black',font=font)
            draw.text((12,34),'Observed / registered expected | red = proposed component | diagnostic only, no physical fault verdict',fill='black',font=font)
            for column,array in enumerate((image,expected)):
                patch=array[max(0,top):min(array.shape[0],bottom),max(0,left):min(array.shape[1],right)]
                patch=cv2.copyMakeBorder(patch,max(0,-top),max(0,bottom-array.shape[0]),max(0,-left),max(0,right-array.shape[1]),cv2.BORDER_CONSTANT,value=(127,127,127))
                patch=cv2.cvtColor(cv2.resize(patch,(340,340),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
                x=column*600+130;y=64;card.paste(Image.fromarray(patch),(x,y))
                box=[x+(l-left)*340/(right-left),y+(t-top)*340/(bottom-top),x+(r-left)*340/(right-left),y+(b-top)*340/(bottom-top)]
                draw.rectangle(box,outline='red',width=3)
            output=OUT/(Path(case['image']).stem+'_'+str(index)+'.png');card.save(output);cards.append(card)
            rows.append(dict(image=case['image'],box=p['box_xyxy'],class_id=p['class_id'],probability=p['confidence'],weak_label_iou=hit,
                weak_matched=hit>=.5,card=str(output),card_sha256=sha(output)))
    assert len(rows)==9 and sum(r['weak_matched'] for r in rows)==2
    for start in range(0,len(cards),3):
        panel=Image.new('RGB',(1200,410*len(cards[start:start+3])),'white')
        for j,card in enumerate(cards[start:start+3]):panel.paste(card,(0,j*410))
        panel.save(OUT/('panel'+str(start//3)+'.png'))
    assert all(sha(Path(p))==v for p,v in pins.items())
    save(OUT/'report.json',dict(status='complete',all_additions=9,weak_matched=2,weak_unmatched=7,cases=rows,pins=pins,
        private_images_not_for_publication=True,no_label_edits=True,no_training=True,no_physical_fault_confirmation=True,field_accuracy=False))
    print(dict(status='complete',cards=len(rows),private_not_published=True),flush=True)


if __name__=='__main__':main()
