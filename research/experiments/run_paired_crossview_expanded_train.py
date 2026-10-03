"""Two representations on extended witnesses; exact cache discipline."""
import copy,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,OUT as CONTEXT,load,save,sha,read_image
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from port_semantic_verifier import fit_head,embeddings as context_embeddings
from paired_boxpool_features import embeddings as footprint_embeddings
from paired_port_semantics import expected_in_source,valid_boxes,paired_features
from paired_shape_features import crop_signature
from paired_crossview_selection import select
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
FOOTPRINT=ROOT/'artifacts/paired_boxpool_original_rows_20261003'
PROPOSALS=ROOT/'artifacts/paired_multimodel_proposal_ceiling_20261003'
OUT=ROOT/'artifacts/paired_crossview_expanded_20261003'


def main():
    if OUT.exists():raise FileExistsError('Preserve expanded joint gate')
    assert load(ROOT/'artifacts/paired_crossview_20261003/crop_report.json')['summary']['qualifies']
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(2);started=time.monotonic();frozen=paired_runtime_fingerprint(REPO)
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_crossview_selection.py'),
        Path(__file__).with_name('paired_boxpool_features.py'),Path(__file__).with_name('port_semantic_verifier.py'),
        ROOT/'artifacts/paired_crossview_expanded_preregistration_20261003/PLAN.md',PROPOSALS/'report.json',
        ROOT/'artifacts/paired_crossview_20261003/crop_report.json')}
    caches=[];dataset=[];heads=[];head_shas=[]
    for root,headroot,kind in ((GEOMETRY,CONTEXT,'context'),(FOOTPRINT,FOOTPRINT,'footprint')):
        prepared=load(root/'features_train/report.json');meta=load(root/'features_train/samples.json')
        for file,key in (('features.pt','aggregate_feature_sha256'),('samples.json','samples_sha256')):
            path=root/'features_train'/file;assert sha(path)==prepared[key];pins[str(path)]=sha(path)
        data=torch.load(root/'features_train/features.pt',map_location='cpu',weights_only=True);dataset.append(data);cache={}
        for i,row in enumerate(meta):
            if row['kind']=='reference_self':continue
            k=tuple(crop_signature(row['box'],s) for s in (1.5,3.)) if kind=='context' else tuple(row['box'])
            cache.setdefault((row['image'],k),data['features'][i])
        caches.append(cache);members=[];digests=[];trained=load(headroot/'heads_oof/report.json')
        for fold in range(3):
            path=headroot/'heads_oof'/f'fold{fold}_head.pt';assert sha(path)==trained['head_pins'][str(path)]
            pins[str(path)]=sha(path);head=torch.nn.Linear(6144,3);head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            head.eval().requires_grad_(False);members.append(head);digests.append(sha(path))
        heads.append(members);head_shas.append(digests)
    OUT.mkdir();save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,context_cache_exact_crop_signature=True,
        footprint_cache_exact_box_only=True,no_reference_self_reuse=True,source_OOF_only=True,
        GT_after_prediction=True,no_deployment=True,field_accuracy=False))
    groups=sorted(load(FOOTPRINT/'features_train/report.json')['source_groups']);assert len(groups)==192
    reference=read_image(DATA/'images/train01/normal_073.JPG');encoder=None;records=[];native_records=[];entries=load(BASE/'train/report.json')['cases']
    try:
        for index,entry in enumerate(entries):
            name=entry['image'];path=PROPOSALS/('train_'+Path(name).stem+'_proposals.json');pins[str(path)]=sha(path);candidate=load(path)
            current=candidate['current'];native=candidate['extended_candidates'];valid_native=[];vectors=[torch.empty((0,6144)),torch.empty((0,6144))];alignment=None;status='no_candidate_exact_short_circuit'
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),phase='OOF_source_features',image=name,completed=index,total=192,seconds=round(time.monotonic()-started,2)))
            if native:
                sp=ORIGINAL_SOURCE=CONTEXT/'features_train'/(Path(name).stem+'_source.json');pins[str(sp)]=sha(sp);source=load(sp)
                imagepath=DATA/'images/train01'/name;assert sha(imagepath)==source['source_sha256'];pins[str(imagepath)]=sha(imagepath);alignment=source['alignment']
                if alignment['alignment_quality']['reliable']:
                    image=read_image(imagepath);expected,mask=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                    valid_native=[native[i] for i in valid_boxes([r['box_xyxy'] for r in native],mask)]
                    for member,extractor in enumerate((context_embeddings,footprint_embeddings)):
                        values=[];missing=[];positions=[]
                        for j,row in enumerate(valid_native):
                            key=tuple(crop_signature(row['box_xyxy'],s) for s in (1.5,3.)) if member==0 else tuple(row['box_xyxy'])
                            cached=caches[member].get((name,key));values.append(cached)
                            if cached is None:positions.append(j);missing.append(row['box_xyxy'])
                        if missing:
                            if encoder is None:
                                import dino_feature_diff as dino
                                encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                            fresh=paired_features(extractor(encoder,image,missing),extractor(encoder,expected,missing))
                            for pos,value in zip(positions,fresh):values[pos]=value
                        vectors[member]=torch.stack(values) if values else torch.empty((0,6144))
                    status='paired_features_ready'
                else:status='registration_abstention'
            fold=groups.index(name)%3
            with torch.inference_mode():probabilities=[heads[m][fold](vectors[m]).softmax(1).tolist() for m in range(2)]
            digests=[head_shas[m][fold] for m in range(2)];trial=select(current,valid_native,probabilities,digests)
            torch.save(dict(context=vectors[0],footprint=vectors[1]),OUT/(Path(name).stem+'_features.pt'))
            save(OUT/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=valid_native,
                member_probabilities=probabilities,head_shas=digests,alignment=alignment,feature_status=status))
            native_records.append(dict(image=name,current=current,proposals=valid_native,vectors=vectors))
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins);oh,nh=matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
            records.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
                gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_crossview_additions'])))
        def evaluate(records):
            totals={kind:{k:sum(r[kind][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for kind in ('current','trial')}
            assert totals['current']==load(GEOMETRY/'full_train/report.json')['summary']['trial']
            normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
            qualifies=totals['trial']['tp']>totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
            return dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,no_deployment=True,field_accuracy=False)
        report=evaluate(records);save(OUT/'source_train_oof.json',report);print(str({k:v for k,v in report.items() if k!='cases'}),flush=True)
        assert {p:sha(Path(p)) for p in pins}==pins and paired_runtime_fingerprint(REPO)==frozen
        if not report['qualifies']:
            save(OUT/'progress.json',dict(status='rejected',phase='source_OOF'));return
        full_foot=fit_head(dataset[1]['features'],dataset[1]['labels']);weight=OUT/'footprint_full_head.pt'
        torch.save(dict(state_dict=full_foot.state_dict(),input_dimensions=6144,fixed_steps=400),weight)
        cp=CONTEXT/'head_full/last_head.pt';pins[str(cp)]=sha(cp);checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
        full_context=torch.nn.Linear(6144,3);full_context.load_state_dict(checkpoint['state_dict']);full_context.eval().requires_grad_(False)
        folder=OUT/'full_train';folder.mkdir();full_records=[];digests=[sha(cp),sha(weight)]
        for entry,native in zip(entries,native_records):
            with torch.inference_mode():probs=[full_context(native['vectors'][0]).softmax(1).tolist(),full_foot(native['vectors'][1]).softmax(1).tolist()]
            trial=select(native['current'],native['proposals'],probs,digests);name=entry['image']
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=native['current'],trial=trial,proposals=native['proposals'],member_probabilities=probs))
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins);oh,nh=matches(native['current']['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
            full_records.append(dict(image=name,current=metric(native['current']['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
                gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_crossview_additions'])))
        full=evaluate(full_records);save(folder/'report.json',full)
        assert {p:sha(Path(p)) for p in pins}==pins and paired_runtime_fingerprint(REPO)==frozen
        save(OUT/'full_pair.json',dict(status='complete',qualifies_train=full['qualifies'],context_head=dict(path=str(cp),sha256=digests[0]),
            footprint_head=dict(path=str(weight),sha256=digests[1]),pins=pins,runtime_fingerprint=frozen,no_validation_training=True,no_deployment=True))
        save(OUT/'progress.json',dict(status='full_train_pass_requires_holdouts' if full['qualifies'] else 'rejected',phase='full_train'));print(str({k:v for k,v in full.items() if k!='cases'}),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise

if __name__=='__main__':main()
