"""Independently replay saved actual evidence plus the memory-isolated final control."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
from paired_graph_hint_link import accepted_native_rows
from verify_paired_geometry_live import parity
OLD=ROOT/'artifacts/paired_pose_native_live_20261004'
LAST=ROOT/'artifacts/paired_pose_native_last_control_retry2_20261004'
SOURCE=ROOT/'artifacts/paired_pose_native_three_20261004'
MEDIAN=ROOT/'artifacts/paired_median_current_head_20261003'
OUT=ROOT/'artifacts/paired_pose_native_live_recovered_20261004'

def main():
    if OUT.exists():raise FileExistsError('Preserve recovered acceptance')
    import numpy as np
    assert load(LAST/'report.json')['status']=='complete'
    protocol=load(OLD/'protocol.json');pins=dict(protocol['pins']);assert {p:sha(Path(p)) for p in pins}==pins
    assert median_runtime_fingerprint(REPO)==protocol['median_runtime_fingerprint']
    rows=load(OLD/'partial.json')['cases'];assert len(rows)==19
    rows=rows+[dict(stage='train',image='disconnected_030.JPG',
        reasons=['ALL_previous_pose_false_cue_control'],report=str(LAST/'initial_report.json'),
        evidence=str(LAST/'native_pose_ports.json'),recovered_final_control=True)]
    assert {(r['stage'],r['image']) for r in rows}=={(r['stage'],r['image']) for r in protocol['categories']}
    cases=[];overrides={}
    for row in rows:
        stage,name=row['stage'],row['image'];report=load(Path(row['report']));evidence=load(Path(row['evidence']))
        original_path=(LAST/'accepted_median_ports.json') if row.get('recovered_final_control') else Path(row['evidence']).with_name('accepted_median_ports.json')
        original=load(original_path);fixed_path=MEDIAN/stage/(Path(name).stem+'_predictions.json');fixed=load(fixed_path)['trial']
        cached_path=SOURCE/('full_train' if stage=='train' else stage)/(Path(name).stem+'_predictions.json')
        expected=load(cached_path)['trial']['paired_semantic_additions']
        for path in (Path(row['report']),Path(row['evidence']),original_path,fixed_path,cached_path,Path(report['inspection']),Path(report['reference'])):pins[str(path)]=sha(path)
        fp=report['image_fingerprints'];assert fp['source_sha256']==sha(Path(report['inspection'])) and fp['reference_sha256']==sha(Path(report['reference'])) and fp['stable_during_visual_analysis']
        for key,value in original.items():
            if key=='supplementary_hints':assert evidence[key][:len(value)]==value
            else:assert evidence[key]==value
        assert not evidence['pose_geometry_policy'].get('fallback_reason')
        added=evidence['supplementary_hints'][len(original['supplementary_hints']):]
        accepted=[];native=evidence.get('pose_geometry_evidence')
        if native:
            parity(fixed['all_predictions'],native['native_current']['all_predictions'])
            parity(expected,native['native']['paired_semantic_additions'])
            accepted=accepted_native_rows(native['native']['paired_semantic_additions'],added,
                    np.asarray(report['alignment']['source_to_reference_homography']),[2736,3648],[2736,3648])
        else:assert not expected and not added
        base={r['image']:r for r in load(BASE/stage/'report.json')['cases']}[name]
        targets=read_targets(stage,name,[2736,3648],base['label_sha256'],pins)
        predictions=fixed['all_predictions']+accepted;old,new=matches(fixed['all_predictions'],targets)[0],matches(predictions,targets)[0]
        actual=dict(row,current=metric(fixed['all_predictions'],targets),trial=metric(predictions,targets),
                    raw_new_cues=len(expected),accepted=len(accepted),gained=sorted(new-old),lost=sorted(old-new))
        if not row.get('recovered_final_control'):
            for key in ('current','trial','raw_new_cues','accepted','gained','lost'):assert actual[key]==row[key],key
        cases.append(actual);overrides[(stage,name)]=actual
    stages={}
    for stage in ('train','inner','outer'):
        source=load(SOURCE/('full_train' if stage=='train' else stage)/'report.json');all_rows=[]
        for row in source['cases']:
            actual=overrides.get((stage,row['image']))
            all_rows.append(dict(image=row['image'],current=row['current'],trial=actual['trial'] if actual else row['current'],
                   gained=actual['gained'] if actual else [],lost=actual['lost'] if actual else []))
        summary={v:{k:sum(r[v][k] for r in all_rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        assert summary['current']==source['summary']['current']
        qualifies=(summary['trial']['tp']>summary['current']['tp'] if stage!='outer' else summary['trial']['tp']>=summary['current']['tp'])
        normal=sum(r['trial']['predictions'] for r in all_rows if r['image'].startswith('normal_'))
        qualifies=qualifies and summary['trial']['unmatched']<=summary['current']['unmatched'] and not any(r['lost'] for r in all_rows) and normal==0
        stages[stage]=dict(qualifies=qualifies,summary=summary,cases=all_rows,normal_cues=normal)
    assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(REPO)==protocol['median_runtime_fingerprint']
    result=dict(status='complete',qualifies=all(s['qualifies'] for s in stages.values()),stages=stages,cases=cases,pins=pins,
        median_runtime_fingerprint=protocol['median_runtime_fingerprint'],head_sha256=protocol['head_sha256'],
        actual_sources=20,unique_dataset_images=270,sam_pending=True,no_deployment=True,field_accuracy=False,
        original_incomplete_run_preserved=True,independent_19_saved_evidence_replay=True,
        final_control_fresh_isolated_after_publication_OOM=True,normal_controls_are_safety_abstentions=True)
    OUT.mkdir(parents=True)
    save(OUT/'report.json',result)
    print({k:v['summary'] for k,v in stages.items()},flush=True)
    assert result['qualifies']

if __name__=='__main__':main()
