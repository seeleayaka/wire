"""Independent source/pixel/proposal/head/weak-score replay for finite trial."""
import hashlib
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from verify_port_exposure_invariance_train import independently_reconstruct
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_native_pose_features import proposals,select
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from inspection_agent.teacher_student_port_support import TEACHER_SHA,STUDENT_SHA
from inspection_agent.feature_residual_port_support import WEIGHT_SHA
from audit_port_multiscale_acceptance import metric,matches
SOURCE=ROOT/'artifacts/photometric_native_proposals_20261004/full'
OUT=ROOT/'artifacts/photometric_native_proposals_replay_20261004'


def main():
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if OUT.exists():raise FileExistsError('Preserve independent photometric trial audit')
    report=load(SOURCE/'report.json')
    assert report['status'] in ('rejected','source_pass_requires_actual_reference_ROI_gates')
    assert report['no_training'] and report['no_deployment'] and not report['field_accuracy']
    frozen=native_pose_runtime_fingerprint(REPO)
    assert frozen==report['runtime']
    assert all(sha(Path(p))==v for p,v in report['pins'].items())
    referencepath=DATA/'images/train01/normal_073.JPG'
    assert sha(referencepath)==REFERENCE_SHA
    reference=read_image(referencepath)
    head=torch.nn.Linear(6144,3)
    assert sha(REPO/HEAD_RELATIVE)==HEAD_SHA
    head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict'])
    head.eval().requires_grad_(False)
    encoder=None
    pins={};counts={};total_candidates=0;total_additions=0
    expected_weights=[TEACHER_SHA,STUDENT_SHA,WEIGHT_SHA]
    for stage,summary in report['stages'].items():
        stage_report=load(SOURCE/stage/'report.json')
        rows=stage_report['cases']
        entries={r['image']:r for r in load(BASE/stage/'report.json')['cases']}
        assert len(rows)==len(entries)=={'train':192,'inner':48,'outer':30}[stage]
        assert {r['image'] for r in rows}==set(entries)
        inferred=0
        for i,row in enumerate(rows):
            name=row['image'];path=SOURCE/stage/(Path(name).stem+'_predictions.json')
            pins[str(path)]=sha(path);case=load(path)
            source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
            assert sha(source)==report['pins'][str(source)]
            currentpath=ROOT/'artifacts/paired_pose_native_three_20261004'/('full_train' if stage=='train' else stage)/(Path(name).stem+'_predictions.json')
            assert case['current']==load(currentpath)['trial']
            policy=case['photometric_policy'];mask=None;image=None;expected=None
            if policy['status']=='skipped':
                assert policy['reason']=='shared_budget_full'
                assert len(case['current']['all_predictions'])==len(case['current']['primary'])+5
            elif case['alignment'].get('alignment_quality',{}).get('reliable'):
                image=read_image(source)
                expected,mask=expected_in_source(reference,case['alignment']['source_to_reference_homography'],image.shape[:2])
                transformed,reconstructed=independently_reconstruct(image,expected,mask)
                assert policy==reconstructed,name
                if policy['status']=='compensated':
                    inferred+=1
                    digest=hashlib.sha256(transformed.tobytes()).hexdigest()
                    assert digest==case['input_pixels_sha256']
                    assert [r['weight_sha256'] for r in case['new_views']]==expected_weights
                    for view in case['new_views']:
                        assert view['source_sha256']==sha(source) and view['input_pixels_sha256']==digest
                        assert view['source_SHA_is_original_coordinate_identity'] and view['input_pixels_changed']
                        assert view['predictions']['source_shape']==list(image.shape[:2])
                    native=proposals(case['new_views'][0],case['new_views'],case['current'])
                    native=[native[j] for j in valid_boxes([r['box_xyxy'] for r in native],mask)]
                    assert native==case['proposals'],name
            else:
                assert policy==dict(status='abstained',reason='unreliable_registration')
            if policy['status']!='compensated':
                assert not case['new_views'] and not case['proposals'] and not case['probabilities']
                assert case['input_pixels_sha256'] is None
            if case['proposals']:
                if encoder is None:
                    import dino_feature_diff as dino
                    encoder=dino._model();encoder.eval().requires_grad_(False)
                torch.set_num_threads(2)
                boxes=[r['box_xyxy'] for r in case['proposals']]
                features=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                with torch.inference_mode():scores=head(features).softmax(dim=1)
                assert torch.allclose(scores,torch.tensor(case['probabilities']),atol=1e-6,rtol=1e-5),name
                for proposal in case['proposals']:
                    assert set(proposal['semantic_model_vote_sha256']).issubset(set(expected_weights))
                total_candidates+=len(boxes)
            assert select(case['current'],case['proposals'],case['probabilities'],HEAD_SHA)==case['trial']
            oldrows=case['current']['all_predictions'];newrows=case['trial']['all_predictions']
            assert newrows[:len(oldrows)]==oldrows
            assert len(case['trial']['primary'])<=5 and len(newrows)<=len(case['trial']['primary'])+5
            total_additions+=len(newrows)-len(oldrows)
            targets=read_targets(stage,name,[2736,3648],entries[name]['label_sha256'],pins)
            assert metric(oldrows,targets)==row['current'] and metric(newrows,targets)==row['trial']
            old=matches(oldrows,targets)[0];new=matches(newrows,targets)[0]
            assert sorted(old-new)==row['lost'] and sorted(new-old)==row['gained']
            if (i+1)%32==0:print(dict(stage=stage,completed=i+1,total=len(rows)),flush=True)
        assert inferred==summary['inferred']
        totals={version:{k:sum(r[version][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
        assert totals==summary['summary']
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies=(totals['trial']['tp']>totals['current']['tp'] if stage!='outer' else totals['trial']['tp']>=totals['current']['tp']) and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in rows) and normal==0
        assert qualifies==summary['qualifies']
        counts[stage]=len(rows)
    assert all(sha(Path(p))==v for p,v in report['pins'].items())
    assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
    OUT.mkdir()
    result=dict(status='pass',experiment_status=report['status'],counts=counts,
        proposals=total_candidates,additions=total_additions,source_report_sha256=sha(SOURCE/'report.json'),
        auditor_sha256=sha(Path(__file__)),independent_pixel_and_score_replay=True,
        original_pixel_DINO_head_scores_recomputed=True,detector_outputs_cached_not_reinferred=True,
        no_deployment=True,not_actual_reference_ROI_or_Qt_acceptance=True,field_accuracy=False)
    save(OUT/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
