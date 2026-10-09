"""Aggregate actual completed evidence, never count fixtures as field accuracy."""
from datetime import datetime,timezone
import argparse
from pathlib import Path
import json
import sys

from core import image_binding,sha256
from run_review import read,save
from run_audit import source_pins
from run_paired_evidence import execute

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/route_topology_continuation_20261005')
    out=parser.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    before=source_pins()
    experiment_files=sorted(p for p in Path(__file__).parent.iterdir() if p.suffix in ['.py','.html','.cjs','.md'])
    pins={str(p):sha256(p) for p in experiment_files}
    save(out/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),
         'mainline_pins':before,'experiment_pins':pins,
         'controls':'identical existing source-mask run on both sides; not new cabinet pairs',
         'real_pair':'fixed original cabinets1/2, crop and verified full-frame context',
         'no_SAM_training_or_inference':True,'no_port_confirmation':True,
         'geometry_minimum_score':.75,'neighborhood_radius_diagonal_ratio':.005,
         'registration_gates':'unchanged mainline'})
    controls=[]
    for name,run in [('full_cabinet_1','bound_sam_topology_20261001/cabinet_1_fresh'),
                     ('crop_cabinet_1','sam_crop_coverage_20261001/cabinet_1_crop_run'),
                     ('crop_cabinet_2','sam_crop_coverage_20261001/cabinet_2_crop_run')]:
        run=ROOT/'artifacts'/run
        result=execute(run,run,out/name)
        assert all(not p['path_shape_difference_detected'] for p in result['proposals'])
        assert not result['automatic_fault_verdict'] and result['new_confirmed_connections']==0
        controls.append({'case':name,'type':'same-image-same-mask-software-self-control',
                         'proposals':len(result['proposals']),'path_difference_nominations':0,
                         'real_new_connections':0,'topology_decision':result['topology_decision']})
    ref=ROOT/'artifacts/sam_crop_coverage_20261001/cabinet_1_crop_run'
    ins=ROOT/'artifacts/sam_crop_coverage_20261001/cabinet_2_crop_run'
    actual_pairs=[]
    for name,context in [('actual_crops',None),('actual_full_context',ROOT/'artifacts/sam_crop_coverage_20261001/frozen_protocol.json')]:
        result=execute(ref,ins,out/name,context)
        assert result['new_confirmed_connections']==0
        actual_pairs.append({'case':name,'matches':result['registration']['matches'],
                             'registration_reliable':result['registration']['alignment_quality']['reliable'],
                             'proposals':len(result['proposals']),'topology_decision':result['topology_decision'],
                             'reason':result.get('reason')})
    tests=read(ROOT/'artifacts/route_topology_20261005_v4/tests.json')
    assert tests['success'] and tests['tests']==87
    browser_paths=[ROOT/'output/playwright/route_topology_20261005_v3/report.json',
                   ROOT/'output/playwright/route_workbench_20261005_v3/report.json']
    browsers=[read(path) for path in browser_paths]
    assert all(r['status']=='passed' and not r['errors'] for r in browsers)
    ui_rechecked=read(ROOT/'artifacts/route_workbench_backend_20261005_v3/rechecked_fixture/report.json')
    ui_draft=read(ROOT/'artifacts/route_workbench_backend_20261005_v3/unconfirmed_fixture/report.json')
    assert ui_rechecked['decision']=='same_visible_terminal_relations'
    assert ui_draft['decision']=='insufficient_evidence'
    OCR=[]
    for relative,expected_calls in [('route_identity_ocr_20261005',20),('route_identity_ocr_tiles_20261005',200)]:
        directory=ROOT/'artifacts'/relative
        report,protocol=read(directory/'report.json'),read(directory/'protocol.json')
        assert report['status']=='complete' and report['new_confirmed_connections']==0
        assert sha256(directory/'protocol.json')==report['protocol_sha256']
        assert all(sha256(path)==digest for path,digest in protocol['models'].items())
        # Identity is independent of image name. Paths in the first frozen
        # protocol are supplied by its fixed-case runner, not presumed in the
        # compact hash/size/frame identity object.
        for case in report['cases']:
            detail=read(directory/case['case']/'report.json')
            assert all(g['confirmed'] is False and g['terminal_assignment'] is None and
                       g['wire_identity'] is None and g['model_observer_count']==1 for g in detail['groups'])
            assert detail['automatic_connections']==[] and detail['confirmed_port_identities']==[]
            assert detail['image_binding']=={k:protocol['sources'][case['case']][k]
                                            for k in ['image_sha256','image_size','coordinate_frame']}
        for source in protocol['sources'].values():
            if 'path' in source:
                assert image_binding(source['path'])=={k:source[k] for k in ['image_sha256','image_size','coordinate_frame']}
        OCR.append({'directory':relative,'report_sha256':sha256(directory/'report.json'),
                    'new_local_calls':expected_calls,'seconds':report['seconds'],'cases':report['cases'],
                    'interpretation':'OCR nominations, NOT authenticated wire/port identity or connection accuracy'})
    after=source_pins()
    assert before==after and all(sha256(path)==digest for path,digest in pins.items())
    save(out/'report.json',{'status':'complete','implementation_stage':'optional_assisted_visible_topology',
         'real_automatic_topology_goal':'not_completed_missing_authentic_visible_terminal_evidence',
         'tests':tests,'browser_checks':sum(len(r['checks']) for r in browsers),
         'browser_reports':{str(p):sha256(p) for p in browser_paths},
         'generated_invariance_trials':{'seed':20261005,'variable_frames':256,'comparisons':768,
                                      'all_passed':True,'evidence_type':'software_generated_masks_not_real_data'},
         'self_controls':controls,'actual_source_pairs':actual_pairs,'OCR_trials':OCR,
         'browser_to_backend_fixture':{'rechecked':'same_visible_terminal_relations','unconfirmed':'insufficient_evidence'},
         'mainline_pins_unchanged':True,'dirty_git_status_unchanged':True,'experiment_pins':pins,
         'no_new_SAM_or_training':True,'new_confirmed_real_connections':0,'new_real_hit_gain':None,
         'cross_cabinet_field_accuracy':None,'new_physical_false_alarms_assessed':None,
         'normal_detections_success_claimed':False,'E_deployed':False,
         'blocker':'Existing cabinet lines often terminate in hidden ducts. OCR is not authenticated; no verified terminal identity/netlist/visible two-ended real case is present. Do not fabricate confirmation.',
         'next_safe_entry':'Offline mask workbench -> reviewed port map -> run_review; real case requires factual terminal identity and expected-edge evidence, not threshold relaxation.'})
    print(json.dumps({'status':'complete','tests':tests['tests'],'browser_checks':26,
         'actual_pairs':actual_pairs,'self_controls':controls,'new_confirmed_real_connections':0,
         'mainline_unchanged':True},ensure_ascii=False))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
