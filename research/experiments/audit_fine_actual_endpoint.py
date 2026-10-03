"""Independently recompute filtered actual cue geometry, prefixes and weak metrics."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from actual_port_native_rows import actual_native_rows
from paired_graph_hint_link import accepted_native_rows
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.teacher_student_port_support import append_verified_student
from inspection_agent.optional_port_crop_review import aligned_predictions
from inspection_agent.context_port_recheck import complete
from inspection_agent.paired_native_pose_features import select
from audit_port_multiscale_acceptance import metric,matches
SOURCE=ROOT/'artifacts/fine_actual_endpoint_20261004'
OUT=ROOT/'artifacts/fine_actual_endpoint_replay_20261004'


def main():
    import numpy as np
    if OUT.exists():raise FileExistsError('Preserve independent actual endpoint replay')
    report=load(SOURCE/'report.json');protocol=load(SOURCE/'protocol.json')
    assert report['status'] in ('rejected','diagnostic_net_gain_requires_full_cohort_and_holdouts')
    assert report['old_source_failure_not_revoked'] and report['no_deployment']
    assert all(sha(Path(p))==v for p,v in report['pins'].items())
    frozen=native_pose_runtime_fingerprint(REPO);assert frozen==report['runtime']
    assert len(report['cases'])==15 and [r['image'] for r in report['cases']]==protocol['selected_train_sources']
    entries={r['image']:r for r in load(BASE/'train/report.json')['cases']};pins={};rows=[]
    for row in report['cases']:
        paths=[Path(row[k]) for k in ('report','baseline_result','trial_result','evidence')]
        for path in paths:pins[str(path)]=sha(path)
        initial,baseline,trial,evidence=[load(p) for p in paths]
        assert initial['image_fingerprints']['source_sha256']==sha(Path(initial['inspection']))
        assert initial['image_fingerprints']['reference_sha256']==sha(Path(initial['reference']))
        old=actual_native_rows(initial,baseline)
        for key,value in baseline.items():
            if key=='supplementary_hints':assert trial[key][:len(value)]==value
            else:assert trial[key]==value
        hints=trial.get('supplementary_hints',[])[len(baseline.get('supplementary_hints',[])):]
        assert len(trial.get('rescue_hints',[]))<=5 and len(trial.get('supplementary_hints',[]))<=5
        matrix=np.asarray(initial['alignment']['source_to_reference_homography'],dtype=float)
        if evidence['selected']:
            for name in ('teacher_student_policy','feature_residual_policy','resolution_policy','paired_geometry_policy','median_geometry_policy','native_pose_policy'):
                assert not baseline.get(name,{}).get('fallback_reason'),(row['image'],name)
            shape=baseline['source_evidence']['predictions']['source_shape']
            reference_shape=baseline['reference_evidence']['predictions']['source_shape']
            current=dict(primary=old[:len(baseline['rescue_hints'])],all_predictions=old)
            digest=evidence['selected'][0]['paired_head_sha256']
            selected=select(current,evidence['proposals'],evidence['probabilities'],digest)['paired_semantic_additions']
            assert selected==evidence['selected']
            refs=[]
            for model in evidence['reference_evidence']:
                refs.extend(aligned_predictions(model['predictions']['merged_predictions'],np.eye(3),reference_shape,reference_shape))
                for record in model['views']:
                    for view in record['views']:
                        for p in view:
                            if p['confidence']>.25 and complete(p,reference_shape):
                                refs.append(dict(zip(('left','top','right','bottom'),p['box_xyxy']),class_id=p['class_id'],
                                    confidence=p['confidence'],valid_warp_fraction=1.,support_tiles=[]))
            replay,added=append_verified_student(baseline,selected,refs,matrix)
            assert len(added)==len(hints)
            for original,actual in zip(added,hints):assert original['box']==actual['box']
            actual_rows=accepted_native_rows(selected,hints,matrix,shape,reference_shape)
            assert actual_rows==evidence['new_actual_rows']
        else:
            assert not hints and not evidence['new_actual_rows'];actual_rows=[]
        targets=read_targets('train',row['image'],[2736,3648],entries[row['image']]['label_sha256'],pins)
        assert metric(old,targets)==row['baseline'] and metric(old+actual_rows,targets)==row['trial']
        oldhits=matches(old,targets)[0];newhits=matches(old+actual_rows,targets)[0]
        assert sorted(newhits-oldhits)==row['gained'] and sorted(oldhits-newhits)==row['lost']
        assert len(actual_rows)==row['added'] and len(evidence['selected'])==row['selected']
        rows.append(row)
    summary={version:{k:sum(r[version][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('baseline','trial')}
    assert summary==report['summary']
    normal=sum(r['added'] for r in rows if r['image'] in protocol['normal_controls'])
    qualifies=summary['trial']['tp']>summary['baseline']['tp'] and summary['trial']['unmatched']<=summary['baseline']['unmatched'] and not any(r['lost'] for r in rows) and normal==0
    assert qualifies==report['qualifies'] and normal==report['normal_added']
    assert all(sha(Path(p))==v for p,v in report['pins'].items()) and all(sha(Path(p))==v for p,v in pins.items())
    assert native_pose_runtime_fingerprint(REPO)==frozen
    OUT.mkdir();result=dict(status='pass',experiment_status=report['status'],sources=15,summary=summary,
        actual_append_reference_ROI_and_forward_link_replayed=True,old_fields_and_budgets_preserved=True,
        source_report_sha256=sha(SOURCE/'report.json'),selected_TRAIN_not_population_accuracy=True,
        no_GUI_SAM_release_acceptance_claimed=True,no_deployment=True,field_accuracy=False)
    save(OUT/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
