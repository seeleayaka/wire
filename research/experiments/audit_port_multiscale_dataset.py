"""Independent geometry/byte replay and train-only visual QA, before full training."""
import json,sys
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
BASE=ROOT/'artifacts/port_training_multiscale_20261002';DATASET=BASE/'dataset';OLD=REPO/'data/derived/port_crop_training_20260929'
RAW=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.port_training_audit import parse_boxes
from inspection_agent.port_tiling import tile_windows
from port_training_multiscale_policy import labels_for_window
from prepare_port_training_multiscale import rect_lines

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    target=BASE/'audit_v2'
    if target.exists():raise FileExistsError('Fresh audit required')
    target.mkdir();manifest=load(DATASET/'dataset_manifest.json');old=load(OLD/'dataset_manifest.json')
    original={r['image']:r for r in old['records']};sources={};counts=Counter();classes=Counter()
    from PIL import Image,ImageDraw
    for r in manifest['records']:
        assert sha(DATASET/r['image'])==r['image_sha256'] and sha(DATASET/r['label'])==r['label_sha256']
        if r['role']=='original_replay':
            p=original[r['image']]
            assert p['image_sha256']==r['image_sha256'] and p['label_sha256']==r['label_sha256'];counts['original_unchanged']+=1
            continue
        assert r['split']=='train'
        name=r['source_image']
        if name not in sources:
            path=RAW/'images/train01'/name;label=RAW/'labels/train01'/(Path(name).stem+'.txt')
            with Image.open(path) as image:w,h=image.size
            boxes=parse_boxes(label.read_text(encoding='utf-8'),w,h)
            ports=[dict(class_id=p['source_class']-3,source_box_index=i,box_xyxy=p['box_xyxy']) for i,p in enumerate(boxes) if p['source_class'] in (3,4)]
            sources[name]=dict(shape=[h,w],ports=ports)
        source=sources[name]
        if r['role']=='multiscale_positive':
            labels=labels_for_window(source['ports'],r['window'],source['shape']);assert labels is not None
            w=r['window'][2]-r['window'][0];h=r['window'][3]-r['window'][1]
            assert rect_lines(labels,w,h)==(DATASET/r['label']).read_text(encoding='utf-8').splitlines()
            classes.update(p['class_id'] for p in labels);counts['positive_geometry_replayed']+=1
        else:
            assert r['role']=='teacher_hard_negative' and r['label_count']==0
            stem=r['image'].split('__hard_t')[0];tile=int(Path(r['image']).stem.split('__hard_t')[1])
            shape=source['shape'];parent=tile_windows(shape[1],shape[0])[tile]
            x,y,right,bottom=r['window'];window=[x+parent[0],y+parent[1],right+parent[0],bottom+parent[1]]
            for port in source['ports']:
                l,t,rr,bb=port['box_xyxy'];assert min(rr,window[2])<=max(l,window[0]) or min(bb,window[3])<=max(t,window[1]),'visible target mislabeled background'
            assert (DATASET/r['label']).read_text().strip()=='';counts['negative_source_annotation_overlap_checked']+=1
    assert counts==dict(original_unchanged=1045,positive_geometry_replayed=123,negative_source_annotation_overlap_checked=64)
    selected=[]
    for size in (640,960):
        for cls in (0,1):
            candidates=[r for r in manifest['records'] if r['role']=='multiscale_positive' and r['size']==size and Path(r['image']).stem.endswith(f'_c{cls}') and
                any(int(line.split()[0])==cls for line in (DATASET/r['label']).read_text().splitlines())]
            candidates.sort(key=lambda r:(r['source_image'],r['image']));selected.append(candidates[0])
    negatives=[r for r in manifest['records'] if r['role']=='teacher_hard_negative'];negatives.sort(key=lambda r:(-r['teacher_score'],r['image']))
    selected+=negatives[:2]
    board=Image.new('RGB',(1500,1100),'#edf0f2')
    for index,r in enumerate(selected):
        with Image.open(DATASET/r['image']) as image:panel=image.convert('RGB')
        w,h=panel.size;draw=ImageDraw.Draw(panel)
        for line in (DATASET/r['label']).read_text().splitlines():
            values=list(map(float,line.split()));cls=int(values[0]);xs=values[1::2];ys=values[2::2]
            draw.rectangle([min(xs)*w,min(ys)*h,max(xs)*w,max(ys)*h],outline='#b6278c' if cls==0 else '#087f8c',width=4)
        if r['role']=='teacher_hard_negative':
            l,t,rr,bb=r['teacher_candidate']['box_xyxy'];x,y,_,_=r['window']
            draw.rectangle([max(0,l-x),max(0,t-y),min(w,rr-x),min(h,bb-y)],outline='#ea8d20',width=4)
        panel.thumbnail((480,480));cell=Image.new('RGB',(500,550),'#edf0f2');cell.paste(panel,((500-panel.width)//2,60))
        label=Path(r['image']).stem
        ImageDraw.Draw(cell).text((8,10),label,fill='#20262b')
        ImageDraw.Draw(cell).text((8,30),'NEGATIVE label; orange=teacher hypothesis' if r['role']=='teacher_hard_negative' else 'TRAIN labels: magenta=plug, cyan=jack',fill='#20262b')
        board.paste(cell,((index%3)*500,(index//3)*550))
    board.save(target/'sample_contact_sheet.jpg',quality=95)
    result=dict(status='complete',counts=dict(counts),new_positive_label_copies_by_class=dict(classes),source_labels_read_only_from_inner_train=True,
        validation_byte_identical=True,sample_images=[r['image'] for r in selected],operator_confirmation=False,rectangles_not_true_masks=True)
    (target/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
