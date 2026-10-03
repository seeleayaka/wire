"""Cheap fixed shape probe, no encoder inference or release writes."""
import copy,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,OUT as ORIGINAL,load,save,sha,read_image
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from paired_shape_features import crop_signature,shaped_features,aspect_examples
from paired_semantic_geometry_examples import training_label
from port_semantic_verifier import crop_tensor,fit_head
from paired_port_semantic_selection import select
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
OUT=ROOT/'artifacts/paired_shape_train_probe_20261003'


def main():
    if OUT.exists():raise FileExistsError('Preserve fixed shape probe')
    import torch
    torch.set_num_threads(1);started=time.monotonic();source=GEOMETRY/'features_train'
    prepared=load(source/'report.json');assert prepared['status']=='complete' and prepared['samples']==3134
    assert sha(source/'features.pt')==prepared['aggregate_feature_sha256'] and sha(source/'samples.json')==prepared['samples_sha256']
    data=torch.load(source/'features.pt',map_location='cpu',weights_only=True);rows=load(source/'samples.json')
    frozen=paired_runtime_fingerprint(REPO);pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_shape_features.py'),
        Path(__file__).with_name('paired_semantic_geometry_examples.py'),Path(__file__).with_name('port_semantic_verifier.py'),
        ROOT/'artifacts/paired_shape_probe_preregistration_20261003/PLAN.md',source/'features.pt',source/'samples.json',source/'report.json')}
    OUT.mkdir();save(OUT/'progress.json',dict(status='running',pid=os.getpid(),phase='crop_identifiability_and_train_geometry'))
    save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,original_samples=3134,train_GT=344,
        input_dimensions=6147,shape_descriptors=['width_over_long_side','height_over_long_side','area_over_long_side_squared'],
        fixed400_steps=True,gate=.98,crop_precision=.98,crop_recall=.25,no_heldout_training=True,
        no_absolute_coordinates_as_features=True,no_DINO_forward=True,no_deployment=True,field_accuracy=False))
    truths={name:[dict(class_id=r['label']-1,box=r['box']) for r in rows if r['image']==name and r['kind']=='gt_port'] for name in prepared['source_groups']}
    features=[shaped_features(data['features'],[r['box'] for r in rows])];metadata=copy.deepcopy(rows);synthetic=[];indices=[];skipped=[]
    signature_collisions=0;changed_labels=0
    for i,row in enumerate(rows):
        if row['kind']!='gt_port':continue
        for box in aspect_examples(row['box']):
            if not all(crop_signature(row['box'],scale)==crop_signature(box,scale) for scale in (1.5,3.)):
                skipped.append(dict(image=row['image'],box=box,reason='floating_crop_signature_differs'));continue
            label,iou=training_label(box,truths[row['image']]);signature_collisions+=1;changed_labels+=label!=row['label']
            indices.append(i);synthetic.append(dict(image=row['image'],box=box,label=label,fold=row['fold'],
                kind='synthetic_aspect',training_GT_maximum_iou=iou,source_sha256=row['source_sha256']))
    assert signature_collisions>0 and changed_labels>0
    first=next(r for r in rows if r['kind']=='gt_port');imagepath=DATA/'images/train01'/first['image'];pins[str(imagepath)]=sha(imagepath);image=read_image(imagepath)
    altered=aspect_examples(first['box'])[0]
    tensor_equal=all(torch.equal(crop_tensor(image,first['box'],s),crop_tensor(image,altered,s)) for s in (1.5,3.));assert tensor_equal
    audit=dict(status='complete',original_GT=sum(r['kind']=='gt_port' for r in rows),identical_crop_variants=signature_collisions,
        contradictory_IoU_training_labels=changed_labels,first_real_source_tensor_exact_at_both_scales=True,
        skipped_different_signatures=skipped,no_encoder_inference=True,no_field_accuracy=True)
    save(OUT/'identifiability_audit.json',audit)
    features.append(shaped_features(data['features'][indices],[r['box'] for r in synthetic]));metadata.extend(synthetic)
    combined=torch.cat(features);labels=torch.tensor([r['label'] for r in metadata]);folds=torch.tensor([r['fold'] for r in metadata])
    torch.save(dict(features=combined,labels=labels,folds=folds),OUT/'features.pt');save(OUT/'samples.json',metadata)
    probabilities=torch.zeros((len(metadata),3));head_pins={}
    for fold in range(3):
        save(OUT/'progress.json',dict(status='running',pid=os.getpid(),phase='fixed_OOF_linear_head',fold=fold,total=3))
        mask=folds==fold
        assert {r['image'] for r in metadata if r['fold']==fold}.isdisjoint({r['image'] for r in metadata if r['fold']!=fold})
        head=fit_head(combined[~mask],labels[~mask]);assert all(torch.isfinite(p).all() for p in head.parameters())
        with torch.inference_mode():probabilities[mask]=head(combined[mask]).softmax(dim=1)
        path=OUT/f'fold{fold}_head.pt';torch.save(head.state_dict(),path);head_pins[str(path)]=sha(path)
    scores,classes=probabilities.max(dim=1);accepted=(scores>=.98)&(classes>0);correct=accepted&(classes==labels)
    gt=torch.tensor([r['kind']=='gt_port' for r in metadata]);tp=int(correct.sum());fp=int((accepted&~correct).sum());gt_correct=int((correct&gt).sum())
    crop=dict(tp=tp,unmatched=fp,precision=tp/max(1,tp+fp),GT_correct=gt_correct,GT_targets=344,recall=gt_correct/344)
    crop['qualifies']=crop['precision']>=.98 and crop['recall']>=.25
    save(OUT/'crop_report.json',dict(status='complete',summary=crop,samples=len(metadata),classifier_OOF_only=True,head_pins=head_pins))
    torch.save(probabilities,OUT/'oof_probabilities.pt');print(str(dict(audit=audit,crop=crop)),flush=True)
    if not crop['qualifies']:
        assert paired_runtime_fingerprint(REPO)==frozen and {p:sha(Path(p)) for p in pins}==pins
        save(OUT/'report.json',dict(status='rejected',stage='crop',summary=crop,pins=pins,no_deployment=True,seconds=round(time.monotonic()-started,2)))
        save(OUT/'progress.json',dict(status='rejected',stage='crop'));return
    records=[];entries=load(BASE/'train/report.json')['cases'];assert len(entries)==192
    for entry in entries:
        name=entry['image'];path=GEOMETRY/'full_train'/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);old=load(path);current=old['trial']
        idx=[i for i,r in enumerate(metadata) if r['image']==name and r['kind']=='novel_weak_proposal']
        native=[metadata[i]['proposal'] for i in idx];values=probabilities[idx].tolist()
        fold=next((metadata[i]['fold'] for i in idx),0);digest=sha(OUT/f'fold{fold}_head.pt')
        trial=select(current,native,values,digest);targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
        oh,nh=matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
        row=dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
            gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_semantic_additions']))
        records.append(row);save(OUT/(Path(name).stem+'_predictions.json'),dict(current=current,trial=trial,proposals=native,probabilities=values,record=row))
    totals={kind:{key:sum(r[kind][key] for r in records) for key in ('tp','unmatched','fn','predictions','targets')} for kind in ('current','trial')}
    assert totals['current']==load(GEOMETRY/'full_train/report.json')['summary']['trial']
    normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
    qualifies=totals['trial']['tp']>totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
    assert paired_runtime_fingerprint(REPO)==frozen and {p:sha(Path(p)) for p in pins}==pins
    save(OUT/'report.json',dict(status='train_source_pass_pending_full_and_holdouts' if qualifies else 'rejected',qualifies_train=qualifies,
        summary=totals,cases=records,crop=crop,normal_cues=normal,pins=pins,head_pins=head_pins,no_deployment=True,
        new_classifier_OOF_only=True,current_geometry_head_was_trained_on_all192=True,field_accuracy=False,seconds=round(time.monotonic()-started,2)))
    save(OUT/'progress.json',dict(status='complete',qualifies_train=qualifies,summary=totals));print(str(dict(qualifies=qualifies,summary=totals)),flush=True)

if __name__=='__main__':main()
