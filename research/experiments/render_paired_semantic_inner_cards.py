"""Every inner addition, not cherry-picked visual proof or label correction."""
import math
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import OUT,DATA,load,save,read_image
from paired_port_semantics import expected_in_source
from audit_port_multiscale_acceptance import overlap
from current_port_baseline_audit import read_targets


def main():
    from PIL import Image,ImageDraw
    source=OUT/'holdouts/inner';report=load(source/'report.json')
    assert report['status']=='complete'
    destination=OUT/'inner_addition_visual_audit';destination.mkdir(exist_ok=False)
    reference=read_image(DATA/'images/train01/normal_073.JPG');tiles=[];records=[];pins={}
    from current_port_baseline_audit import BASE
    entries={r['image']:r for r in load(BASE/'inner/report.json')['cases']}
    for case in report['cases']:
        if not case['additions']:continue
        name=case['image'];saved=load(source/(Path(name).stem+'_predictions.json'))
        image=read_image(DATA/'images/train01'/name)
        expected,_=expected_in_source(reference,saved['alignment']['source_to_reference_homography'],image.shape[:2])
        targets=read_targets('inner',name,[2736,3648],entries[name]['label_sha256'],pins)
        for addition in saved['trial']['paired_semantic_additions']:
            l,t,r,b=addition['box_xyxy'];cx,cy=(l+r)/2,(t+b)/2;side=max(320.,max(r-l,b-t)*4)
            window=[math.floor(cx-side/2),math.floor(cy-side/2),math.ceil(cx+side/2),math.ceil(cy+side/2)]
            maximum=max((overlap(addition['box_xyxy'],row['box']) for row in targets if row['class_id']==addition['class_id']),default=0.)
            records.append(dict(image=name,class_id=addition['class_id'],maximum_same_class_GT_iou=maximum,
                probability=addition['paired_semantic_probability'],detector_score=addition['proposal_detector_score']))
            tile=Image.new('RGB',(980,550),'white');draw=ImageDraw.Draw(tile)
            draw.text((8,8),f'{name}  class={addition["class_id"]}  GT IoU={maximum:.4f}  p={addition["confidence"]:.4f}',fill='black')
            for index,array in enumerate((image,expected)):
                panel=Image.fromarray(array[:,:,::-1]).crop(window)
                canvas=ImageDraw.Draw(panel)
                def rectangle(box,color,width):
                    x1,y1,x2,y2=box
                    canvas.rectangle((x1-window[0],y1-window[1],x2-window[0],y2-window[1]),outline=color,width=width)
                if index==0:
                    for row in targets:rectangle(row['box'],'#aaaaaa',2)
                    for row in saved['current']['all_predictions']:rectangle(row['box_xyxy'],'#33aaff',2)
                rectangle(addition['box_xyxy'],'red',3)
                panel.thumbnail((480,480));tile.paste(panel,(8+index*490,60))
                draw.text((8+index*490,30),'OBSERVED: red=new / blue=current / gray=GT' if index==0 else 'ALIGNED EXPECTED: red=same source window',fill='black')
            tiles.append(tile)
    assert len(tiles)==sum(r['additions'] for r in report['cases'])
    sheet=Image.new('RGB',(980,550*len(tiles)),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,(0,i*550))
    path=destination/'all_inner_additions.jpg';sheet.save(path,quality=94)
    save(destination/'geometry_diagnostics.json',dict(status='complete',all_additions_reviewable=True,cases=records,
        GT_unchanged=True,physical_fault_verdict=False,metric_threshold_unchanged=True))
    print(str(path));print(str(records))


if __name__=='__main__':main()
