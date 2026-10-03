"""Fixed three-fold source-group semantic feasibility, no holdouts/deployment."""
import json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/port_semantic_verifier_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(HF_HUB_OFFLINE='1',TORCH_HOME=str(OUT/'torch_cache'))
from inspection_agent.optional_port_crop_review import sha,read_image
from port_semantic_verifier import embeddings,coverage,context_box,fit_head,precision_gate,PROBABILITY_GATE,MIN_PRECISION,MIN_RECALL
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    if OUT.exists():raise FileExistsError('Preserve experiment')
    groupfile=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json';groups=load(groupfile)
    names=sorted(groups['train_sources']);assert len(names)==192 and set(names).isdisjoint(groups['inner_val_sources'])
    weight=REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth'
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_semantic_verifier.py'),groupfile,weight,
        REPO/'prototype/dino_feature_diff.py',REPO/'models/dinov2/dinov2/models/vision_transformer.py')}
    for name in names:
        for p in (DATA/'images/train01'/name,DATA/'labels/train01'/(Path(name).stem+'.txt')):pins[str(p)]=sha(p)
    OUT.mkdir()
    save(OUT/'protocol.json',dict(pins=pins,train_sources=names,whole_source_fold='Sorted192 source index modulo3',
        frozen_dino=True,features='L2-normalized CLS and center4x4 patchmean; square224 crops at1.5x and3x;1536 dims',
        no_filename_or_coordinates_as_features=True,classes=['other','unplugged_plug','unplugged_jack'],
        positive_crops='All training GT3/4, not detector evaluation',
        negative_crops='At most2 biggest GT0/1/2 per source whose3x square context has no port overlap, plus2 fixed128px grid distractors whose3x context has no GT overlap',
        negative_annotation_caveat=True,head='400 fixed full-batch AdamW steps lr.01 decay.001; inverse class weights',
        probability_gate=PROBABILITY_GATE,min_precision=MIN_PRECISION,min_recall=MIN_RECALL,
        feasibility_before_detection_evaluation=True,no_validation_images=True,production_changed=False,field_accuracy=False))
    import torch
    import dino_feature_diff as dino
    torch.set_num_threads(2);model=dino._model();torch.set_num_threads(2);model.requires_grad_(False)
    started=time.monotonic();all_features=[];all_labels=[];folds=[];metadata=[]
    try:
        for index,name in enumerate(names):
            image=read_image(DATA/'images/train01'/name);h,w=image.shape[:2];rows=[]
            for line in (DATA/'labels/train01'/(Path(name).stem+'.txt')).read_text(encoding='utf-8').splitlines():
                cls,cx,cy,bw,bh=map(float,line.split());box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
                if box[2]>box[0] and box[3]>box[1]:rows.append(dict(class_id=int(cls),box=box))
            ports=[r for r in rows if r['class_id'] in (3,4)]
            positives=[dict(box=r['box'],label=r['class_id']-2,kind='gt_port') for r in ports]
            negative_rows=sorted([r for r in rows if r['class_id'] in (0,1,2) and
                not any(coverage(context_box(r['box'],3.),p['box'])>.1 for p in ports)],key=lambda r:-(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]))[:2]
            examples=positives+[dict(box=r['box'],label=0,kind='gt_other') for r in negative_rows]
            added=0
            for fy,fx in ((.15,.15),(.15,.5),(.15,.85),(.5,.15),(.5,.5),(.5,.85),(.85,.15),(.85,.5),(.85,.85)):
                box=[fx*w-64,fy*h-64,fx*w+64,fy*h+64]
                # Context must not hide an annotated port outside the tight box.
                context=[fx*w-192,fy*h-192,fx*w+192,fy*h+192]
                if any(coverage(context,r['box'])>.1 for r in rows):continue
                examples.append(dict(box=box,label=0,kind='grid_distractor'));added+=1
                if added>=2:break
            features=embeddings(model,image,[r['box'] for r in examples]);assert features.shape==(len(examples),1536)
            all_features.append(features);all_labels.extend(r['label'] for r in examples);folds.extend([index%3]*len(examples))
            metadata.extend(dict(image=name,fold=index%3,**r) for r in examples)
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),phase='frozen_features',completed=index+1,total=len(names),
                samples=len(metadata),seconds=round(time.monotonic()-started,2)))
            print(f'features {index+1}/192 samples={len(metadata)}',flush=True)
        features=torch.cat(all_features);labels=torch.tensor(all_labels);fold_ids=torch.tensor(folds)
        assert torch.isfinite(features).all() and not any(p.requires_grad for p in model.parameters())
        torch.save(dict(features=features,labels=labels,folds=fold_ids),OUT/'features.pt');save(OUT/'samples.json',metadata)
        predictions=torch.zeros((len(labels),3));fold_reports=[]
        for fold in range(3):
            mask=fold_ids==fold;assert {m['image'] for m in metadata if m['fold']==fold}.isdisjoint(
                m['image'] for m in metadata if m['fold']!=fold)
            head=fit_head(features[~mask],labels[~mask])
            with torch.inference_mode():predictions[mask]=head(features[mask]).softmax(dim=1)
            torch.save(head.state_dict(),OUT/f'fold{fold}_head.pt')
            fold_reports.append(dict(fold=fold,**precision_gate(predictions[mask],labels[mask])))
        overall=precision_gate(predictions,labels);torch.save(predictions,OUT/'oof_probabilities.pt')
        assert {p:sha(Path(p)) for p in pins}==pins
        save(OUT/'report.json',dict(status='complete',qualifies_crop_feasibility=overall['qualifies'],
            overall=overall,folds=fold_reports,samples=len(metadata),counts=torch.bincount(labels,minlength=3).tolist(),
            seconds=round(time.monotonic()-started,2),detector_success_not_evaluated=True,
            no_validation_evaluation=True,field_accuracy=False,production_changed=False))
        save(OUT/'progress.json',dict(status='complete',qualifies_crop_feasibility=overall['qualifies'],overall=overall))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise
if __name__=='__main__':main()
