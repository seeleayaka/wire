"""All192 training sources retained; teacher mining restricted to training empty crops."""
import copy,json,os,shutil,sys,time
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/port_training_multiscale_20261002'
OLD=REPO/'data/derived/port_crop_training_20260929'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
WEIGHT=REPO/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
sys.path.insert(0,str(REPO))
from port_training_multiscale_policy import POLICY,selected_seeds,positive_window,hard_negative_selection
from inspection_agent.optional_port_crop_review import sha,read_image

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def rect_lines(labels,width,height):
    lines=[]
    for row in labels:
        l,t,r,b=row['box_xyxy_local'];assert 0<=l<r<=width and 0<=t<b<=height
        l,t,r,b=l/width,t/height,r/width,b/height
        lines.append(f"{row['class_id']} {l:.9f} {t:.9f} {r:.9f} {t:.9f} {r:.9f} {b:.9f} {l:.9f} {b:.9f}")
    return lines

def add_crop(dataset,name,image,labels,source_name,role,window,extras):
    from PIL import Image
    im=dataset/'images/train'/(name+'.jpg');lab=dataset/'labels/train'/(name+'.txt')
    assert not im.exists() and not lab.exists()
    width,height=image.size;image.convert('RGB').save(im,quality=95,subsampling=0)
    lines=rect_lines(labels,width,height);lab.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
    with Image.open(im) as reread:assert reread.size==(width,height)
    assert lab.read_text(encoding='utf-8').splitlines()==lines
    return dict(source_image=source_name,split='train',image=im.relative_to(dataset).as_posix(),
        label=lab.relative_to(dataset).as_posix(),image_sha256=sha(im),label_sha256=sha(lab),label_count=len(labels),
        role=role,window=window,**extras)

def main():
    if OUT.exists():raise FileExistsError('Fresh training artifact directory required')
    (OUT/'config/Ultralytics').mkdir(parents=True);dataset=OUT/'dataset'
    for directory in ('images/train','images/val','labels/train','labels/val'):(dataset/directory).mkdir(parents=True)
    manifest=load(OLD/'dataset_manifest.json');assert manifest['status']=='complete' and manifest['source_group_disjoint']
    train_names=sorted({r['source_image'] for r in manifest['records'] if r['split']=='train'})
    val_names=sorted({r['source_image'] for r in manifest['records'] if r['split']=='val'})
    assert len(train_names)==192 and len(val_names)==48 and set(train_names).isdisjoint(val_names)
    assert sha(WEIGHT)==load(REPO/'config/port_crop_calibration_frozen_20260930.json')['fingerprints']['weight']
    save(OUT/'protocol.json',dict(policy=POLICY,train_sources=train_names,inner_val_sources=val_names,
        teacher_weight_sha256=sha(WEIGHT),base_manifest_sha256=sha(OLD/'dataset_manifest.json'),
        policy_sha256=sha(Path(__file__).with_name('port_training_multiscale_policy.py')),
        annotations_used_only_for_training_augmentation=True,heldout_selection_unchanged=True,
        outer_val_test_used=False,negative_means_no_class3_or4_annotation_not_assembly_normal=True,
        training_recipe=dict(initialization='accepted teacher',epochs=2,imgsz=960,batch=1,nbs=16,freeze=10,
          optimizer='SGD',lr0=.0005,lrf=.1,warmup_epochs=.5,seed=20261002,
          mosaic=0,scale=.05,translate=.02,flips=0,select_final_last_checkpoint=True)))
    from PIL import Image
    from inspection_agent.port_training_audit import parse_boxes
    records=[];protected={};sources={};attempted=0;skipped=0;seen_windows=set()
    for old in manifest['records']:
        for kind in ('image','label'):
            source=OLD/old[kind];assert sha(source)==old[kind+'_sha256']
            shutil.copy2(source,dataset/old[kind])
        copied=copy.deepcopy(old);copied['role']='original_replay';records.append(copied)
    # No new heldout source image/label reads in sample design. Validation crops are unchanged copies.
    for name in train_names:
        path=DATA/'images/train01'/name;label=DATA/'labels/train01'/(Path(name).stem+'.txt')
        protected[str(path)]=sha(path);protected[str(label)]=sha(label)
        with Image.open(path) as image:
            width,height=image.size;boxes=parse_boxes(label.read_text(encoding='utf-8'),width,height)
            ports=[dict(class_id=b['source_class']-3,source_box_index=i,box_xyxy=b['box_xyxy']) for i,b in enumerate(boxes) if b['source_class'] in (3,4)]
            sources[name]=dict(source_sha256=sha(path),label_sha256=sha(label),port_count=len(ports),shape=[height,width])
            for seed in selected_seeds(name,ports):
                for size in POLICY['sizes']:
                    attempted+=1;planned=positive_window(seed,ports,size,[height,width])
                    if planned is None:skipped+=1;continue
                    key=(name,*planned['window'])
                    if key in seen_windows:continue
                    seen_windows.add(key);stem=Path(name).stem+f"__aug_s{size}_c{seed['class_id']}"
                    record=add_crop(dataset,stem,image.crop(tuple(planned['window'])),planned['labels'],name,'multiscale_positive',planned['window'],
                        dict(source_sha256=sha(path),source_label_sha256=sha(label),seed_source_box_index=seed['source_box_index'],size=size))
                    records.append(record)
    save(OUT/'positive_plan.json',dict(attempted=attempted,skipped_unsafe=skipped,sources=sources,
        records=[r for r in records if r['role']=='multiscale_positive']))
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);torch.manual_seed(20261002);model=YOLO(str(WEIGHT))
    negatives=[r for r in manifest['records'] if r['split']=='train' and r['label_count']==0]
    assert len(negatives)==191;mined=[];start=time.monotonic()
    for index,record in enumerate(negatives,1):
        path=OLD/record['image'];image=read_image(path);h,w=image.shape[:2]
        assert (OLD/record['label']).read_text(encoding='utf-8').strip()==''
        output=model.predict(image,imgsz=960,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False)[0]
        torch.set_num_threads(4);rows=[]
        for b in output.boxes:
            l,t,r,bottom=map(float,b.xyxy[0].tolist())
            if l>16 and t>16 and r<w-16 and bottom<h-16:
                rows.append(dict(box_xyxy=[l,t,r,bottom],class_id=int(b.cls.item()),confidence=float(b.conf.item())))
        rows.sort(key=lambda r:(-r['confidence'],*r['box_xyxy'],r['class_id']))
        if rows:
            best=rows[0];l,t,r,b=best['box_xyxy'];size=640
            x=max(0,min(w-size,round((l+r)/2-size/2)));y=max(0,min(h-size,round((t+b)/2-size/2)))
            mined.append(dict(source_image=record['source_image'],tile_id=record['tile_id'],teacher_score=best['confidence'],
                teacher_candidate=best,base_record=record,window_local=[x,y,x+size,y+size]))
        if index%10==0 or index==len(negatives):
            save(OUT/'mining_progress.json',dict(completed=index,total=len(negatives),seconds=round(time.monotonic()-start,2)))
            print(f'hard-negative mining {index}/{len(negatives)}',flush=True)
    selected=hard_negative_selection(mined);save(OUT/'hard_negative_plan.json',dict(all_candidates=mined,selected=selected))
    for item in selected:
        old=item['base_record'];assert sha(OLD/old['image'])==old['image_sha256']
        with Image.open(OLD/old['image']) as image:
            stem=Path(item['source_image']).stem+f"__hard_t{item['tile_id']:02}"
            record=add_crop(dataset,stem,image.crop(tuple(item['window_local'])),[],item['source_image'],'teacher_hard_negative',item['window_local'],
                dict(base_image_sha256=old['image_sha256'],teacher_score=item['teacher_score'],teacher_candidate=item['teacher_candidate']))
            records.append(record)
    assert {path:sha(Path(path)) for path in protected}==protected
    assert sha(WEIGHT)==load(OUT/'protocol.json')['teacher_weight_sha256']
    assert {r['source_image'] for r in records if r['split']=='train'}==set(train_names)
    assert {r['source_image'] for r in records if r['split']=='val'}==set(val_names)
    for record in records:
        assert sha(dataset/record['image'])==record['image_sha256'] and sha(dataset/record['label'])==record['label_sha256']
    yaml=dataset/'data.yaml';yaml.write_text('path: '+dataset.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n',encoding='utf-8')
    result=dict(status='complete',records=records,source_split='train01',train_sources=192,inner_val_sources=48,
        source_group_disjoint=True,train_crops=sum(r['split']=='train' for r in records),inner_val_crops=576,
        roles=dict(Counter(r['role'] for r in records if r['split']=='train')),positive_attempts=attempted,unsafe_positive_skips=skipped,
        teacher_weight_sha256=sha(WEIGHT),original_validation_crops_unchanged=True,outer_val_test_used=False,
        training_started=False,rectangles_not_true_masks=True,protected_source_hashes=protected)
    save(dataset/'dataset_manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','protected_source_hashes')},indent=2),flush=True)

if __name__=='__main__':main()
