"""Real SAM-output repair check plus 30-case archived prediction equivalence.

Does not rerun SAM/DINO, train, tune, or mutate saved tasks and baseline reports.
"""
import os
import sys
import copy
import json
import hashlib
import unittest
from pathlib import Path

sys.dont_write_bytecode=True
REPO=Path('E:/PythonProject10')
WORK=Path(__file__).resolve().parents[1]
OUT=WORK/'artifacts/port_bridge_fix_20261002'
LIVE=WORK/'artifacts/main_scenario_live_20261002'
ACCEPT=WORK/'artifacts/port_crop_acceptance_20260930'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'prototype'))
os.environ.update(QT_QPA_PLATFORM='offscreen',YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
(OUT/'config/Ultralytics').mkdir(parents=True,exist_ok=True)
import torch  # must precede Qt on this host
from inspection_agent.port_crop_gui_bridge import run_gui_port_review,render_port_overlay
from inspection_agent.optional_port_crop_review import SCENE,REFERENCE_SHA

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bounds(parent):return parent.get('bbox_xyxy') or [parent[k] for k in ('left','top','right','bottom')]
def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.

def main():
    tests=unittest.TestSuite()
    for pattern in ('test_port_crop_gui_bridge.py','test_port_state_hint.py','test_port_tiling.py',
                    'test_inspection_agent.py','test_inspection_agent_gui_contract.py',
                    'test_local_evidence_bridge.py','test_local_review_gui_bridge.py'):
        tests.addTests(unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern=pattern))
    test_result=unittest.TextTestRunner(verbosity=1).run(tests)
    save(OUT/'regression.json',dict(tests=test_result.testsRun,failures=len(test_result.failures),errors=len(test_result.errors)))
    if not test_result.wasSuccessful():raise RuntimeError('Regression failed')

    real=[]
    for case in load(LIVE/'report.json')['cases']:
        source=Path(case['output']);target=OUT/case['image'].split('.')[0];target.mkdir(exist_ok=True)
        original=load(source/'report.json');before=copy.deepcopy(original)
        task_sha=sha(Path(case['task']));report_sha=sha(source/'report.json')
        result=run_gui_port_review(original,project=REPO,enabled=True,scene=SCENE)
        save(target/'port_review.json',result)
        assert result['parents']==original['review_regions'] and original==before
        assert result['fallback_reason']!='ValueError: invalid parent geometry'
        assert len(result['tile_hints'])<=1 and not result['automatic_fault_verdict']
        if result['status']=='applied':render_port_overlay(source/'aligned.jpg',result,target/'overlay.jpg')
        else:assert result['fallback_reason']=='local_alignment_not_supported',result['fallback_reason']
        # Labels from the completed prior visual run, used only after port selection.
        labels=load(source/'label_audit.json')['aligned_targets']
        parent_boxes=[bounds(p) for p in original['review_regions']]
        combined=parent_boxes+[bounds(h['box']) for h in result['tile_hints']]
        metric=lambda boxes:dict(any_overlap=sum(any(iou(t,b)>0 for b in boxes) for t in labels),
            iou050=sum(max((iou(t,b) for b in boxes),default=0)>=.5 for t in labels))
        assert task_sha==sha(Path(case['task'])) and report_sha==sha(source/'report.json')
        row=dict(image=case['image'],status=result['status'],reason=result['fallback_reason'],
                 new_hints=len(result['tile_hints']),targets=len(labels),before=metric(parent_boxes),after=metric(combined),
                 original_report_unchanged=True,original_task_unchanged=True,new_port_inference=result['status']=='applied')
        real.append(row);print('LIVE '+json.dumps(row),flush=True)

    old=load(REPO/'output/port_state_hints_validation_20260929/report.json')
    expected={c['image']:c for c in load(ACCEPT/'partial_cases.json')}
    replay=[]
    for prior in sorted(old['cases'],key=lambda c:c['image']):
        cached=load(ACCEPT/'cache'/('val01_'+Path(prior['image']).stem+'.json'))
        base=dict(reference=str(DATA/'train01/normal_073.JPG'),inspection=str(DATA/'val01'/prior['image']),
                  review_regions=copy.deepcopy(prior['parents']),existing_port_hints=copy.deepcopy(prior['hints']),
                  alignment_quality={'reliable':True},alignment={'source_to_reference_homography':prior['actual_homography']},
                  local_alignment=[],image_fingerprints=dict(source_sha256=cached['source_sha256'],reference_sha256=REFERENCE_SHA,stable_during_visual_analysis=True))
        variants=[]
        for sam_shape in (False,True):
            report=copy.deepcopy(base)
            if sam_shape:
                report['review_regions']=[{**{k:v for k,v in p.items() if k not in ('left','top','right','bottom')},
                    'bbox_xyxy':bounds(p)} for p in report['review_regions']]
            unchanged=copy.deepcopy(report)
            result=run_gui_port_review(report,project=REPO,enabled=True,scene=SCENE,
                prediction_provider=lambda image,c=cached:copy.deepcopy(c))
            assert result['status']=='applied',result['fallback_reason']
            assert report==unchanged and result['parents']==report['review_regions']
            for key in ('existing_hints','tile_hints','aligned_predictions','selection_audit'):
                assert result[key]==expected[prior['image']][key],(prior['image'],key)
            variants.append(result)
        replay.append(dict(image=prior['image'],formats_equivalent=True,new_hints=len(variants[0]['tile_hints'])))
    assert len(replay)==30 and sum(c['new_hints'] for c in replay)==6
    save(OUT/'verification.json',dict(regression_tests=test_result.testsRun,real_sam_reports=real,
        archived_val01_replay=replay,archived_replay_new_visual_inference=False,
        archived_replay_new_port_inference=False,sam_shaped_variant_is_schema_fixture=True,
        training_or_tuning=False,default_port_enabled=False,repair_reinspection_performed=False,
        formal_source_sha=sha(REPO/'inspection_agent/port_crop_gui_bridge.py')))
    print('VERIFIED '+str(OUT/'verification.json'),flush=True)

if __name__=='__main__':main()
