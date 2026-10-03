"""ALL270 cached committee and actual teacher/student evidence component gate."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from paired_semantic_components import unique_components
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_multiscale_acceptance import metric,matches
SOURCE=ROOT/'artifacts/paired_semantic_committee_20261003'
DETECTORS=ROOT/'artifacts/paired_support_graph_20261003/source_selections'
OUT=ROOT/'artifacts/paired_semantic_components_20261003'


def main():
    if OUT.exists():raise FileExistsError('Preserve component preflight')
    protocol=load(SOURCE/'protocol.json');pins=dict(protocol['pins']);frozen=resolution_runtime_fingerprint(REPO)
    assert frozen==protocol['runtime_fingerprint'] and {p:sha(Path(p)) for p in pins}==pins
    for p in (Path(__file__),Path(__file__).with_name('paired_semantic_components.py'),ROOT/'artifacts/paired_semantic_component_preregistration_20261003/PLAN.md'):
        pins[str(p)]=sha(p)
    OUT.mkdir();save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,cached_source_only=True,
        exact_distinct_detection_components=True,cross_head_same_class_iou=.5,original_floor=.05,
        old_prefix_and_shared5plus5_preserved=True,no_GT_component_inputs=True,no_automatic_deployment=True,
        training_committee_not_OOF=True,validation_reused=True,field_accuracy=False))
    summary={};results=[]
    for stage,count in (('train',192),('inner',48),('outer',30)):
        indexpath=DETECTORS/stage/'index.json';pins[str(indexpath)]=sha(indexpath);index=load(indexpath)
        detectors={r['image']:r for r in index['records']};baselinepath=BASE/stage/'report.json';pins[str(baselinepath)]=sha(baselinepath)
        entries=load(baselinepath)['cases'];assert len(entries)==len(detectors)==count
        folder=OUT/stage;folder.mkdir();records=[]
        for entry in entries:
            name=entry['image'];path=SOURCE/stage/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);source=load(path)
            path=Path(detectors[name]['path']);assert sha(path)==detectors[name]['sha256'];pins[str(path)]=sha(path);models=load(path)
            current=source['current'];candidates=source['trial']['paired_committee_additions']
            accepted,audit=unique_components(candidates,models['teacher'],models['student'])
            selected=current['all_predictions']+accepted
            assert selected[:len(current['all_predictions'])]==current['all_predictions'] and len(selected)<=len(current['primary'])+5
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial_native=selected,
                candidates=candidates,accepted=accepted,component_audit=audit,alignment=source['alignment']))
            targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
            old=current['all_predictions'];oh,nh=matches(old,targets)[0],matches(selected,targets)[0]
            records.append(dict(image=name,current=metric(old,targets),trial=metric(selected,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                candidates=len(candidates),accepted=len(accepted),suppressed=len(candidates)-len(accepted)))
        totals={v:{k:sum(row[v][k] for row in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
        normal=sum(row['trial']['predictions'] for row in records if row['image'].startswith('normal_'))
        gain=totals['trial']['tp']>=totals['current']['tp'] if stage=='outer' else totals['trial']['tp']>totals['current']['tp']
        qualifies=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(row['lost'] for row in records) and normal==0
        result=dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,
            no_automatic_deployment=True,fresh_source_reference_ROI_Qt_SAM_pending=True,validation_reused=True,field_accuracy=False)
        save(folder/'report.json',result);summary[stage]=totals;results.append(qualifies);print(str(dict(stage=stage,qualifies=qualifies,summary=totals)),flush=True)
    assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
    save(OUT/'report.json',dict(status='complete',qualifies=all(results),summary=summary,pins=pins,
        no_GT_components=True,cached_source_only=True,fresh_source_reference_ROI_Qt_SAM_pending=True,
        no_automatic_deployment=True,validation_reused=True,field_accuracy=False))


if __name__=='__main__':main()
