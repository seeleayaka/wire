"""Frozen portable backend replay over ALL actual prospective and safety sources."""
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT/'staging/native_pose_acceptance_project'
OUT=ROOT/'artifacts'/os.environ.get('WIRE_NATIVE_ALL_OUT','native_portable_all_actual_20261004')
LIVE=ROOT/'artifacts/paired_pose_native_live_recovered_20261004/report.json'
SAM=ROOT/'artifacts/native_portable_complete_sam_20261004_v3/acceptance.json'
sys.path[:0]=[str(PROJECT),str(PROJECT/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
                 YOLO_CONFIG_DIR=str(OUT/'config'))
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)

def main():
    if OUT.exists():raise FileExistsError('Preserve portable final workflow replay')
    sam=load(SAM);assert sam['status']=='complete' and sam['new_inspection_sam'] and sam['actual_qt_button_worker_callback']
    live=load(LIVE);assert live['qualifies'] and len(live['cases'])==20
    import psutil
    assert psutil.virtual_memory().available>6*2**30,'Wait for SAM and memory-intensive processes to exit'
    (OUT/'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    import torch
    torch.set_num_threads(2)
    from inspection_agent import paired_native_pose as backend
    from inspection_agent.optional_port_crop_review import sha
    from paired_graph_hint_link import accepted_native_rows
    from audit_port_multiscale_acceptance import metric
    from current_port_baseline_audit import BASE,read_targets
    import numpy as np
    assert Path(backend.__file__).is_relative_to(PROJECT)
    frozen=backend.native_pose_runtime_fingerprint(PROJECT)
    pins={str(path):sha(path) for path in (Path(__file__),LIVE,SAM)}
    started=time.monotonic();cases=[]
    for index,row in enumerate(live['cases']):
        save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=index,total=20,stage=row['stage'],image=row['image']))
        report=load(row['report']);actual=load(row['evidence']);folder=Path(row['evidence']).parent
        original_path=folder/'accepted_median_ports.json';original=load(original_path);protected=copy.deepcopy(original)
        result=backend.append_native_pose_review(report,original,project=PROJECT)
        assert result['native_pose_policy'].get('fallback_reason') is None,result['native_pose_policy']
        assert original==protected
        for key,value in original.items():
            if key=='supplementary_hints':assert result[key][:len(value)]==value
            else:assert result[key]==value
        new=result['supplementary_hints'][len(original['supplementary_hints']):]
        expected=actual['supplementary_hints'][len(original['supplementary_hints']):]
        # Only deliberate policy identifiers and the Median->Native display title
        # change. The safety disclaimer, boxes, scores and model SHA remain exact.
        suffix='Reference non-detection does not prove physical fault or continuity.'
        assert all(h.get('warning','').endswith(suffix) for h in new+expected)
        normalize=lambda hints:[{k:v for k,v in h.items() if k not in ('native_pose_policy_id','median_geometry_policy_id','warning')} for h in hints]
        assert normalize(new)==normalize(expected),'Portable native cue differs on '+row['image']
        for path in (Path(row['report']),Path(row['evidence']),original_path):pins[str(path)]=sha(path)
        evidence=result.get('native_pose_evidence');accepted=[]
        if evidence:
            accepted=accepted_native_rows(evidence['native']['paired_semantic_additions'],new,
                np.asarray(report['alignment']['source_to_reference_homography']),[2736,3648],[2736,3648])
        else:assert not new
        stage=row['stage'];name=row['image'];base=next(r for r in load(BASE/stage/'report.json')['cases'] if r['image']==name)
        targets=read_targets(stage,name,[2736,3648],base['label_sha256'],pins)
        fixed=load(ROOT/'artifacts/paired_median_current_head_20261003'/stage/(Path(name).stem+'_predictions.json'))['trial']
        scored=metric(fixed['all_predictions']+accepted,targets);assert scored==row['trial']
        target=OUT/stage/(Path(name).stem+'_portable_result.json');save(target,result)
        case=dict(stage=stage,image=name,added=len(new),metrics=scored,full_field_prefix_preserved=True,
                  exact_final_cue_parity=True,evidence=str(target));cases.append(case)
        save(OUT/'partial.json',dict(cases=cases));print(json.dumps(case),flush=True)
    assert backend.native_pose_runtime_fingerprint(PROJECT)==frozen and {p:sha(Path(p)) for p in pins}==pins
    result=dict(status='complete',all_20_actual_sources=True,portable_final_cues_exactly_match=True,
                preserves_270_image_development_totals=True,source_tp=[295,68,40],source_unmatched=[4,0,1],
                cases=cases,pins=pins,runtime_fingerprint=frozen,sam_qt_acceptance_sha256=sha(SAM),
                default_off=True,manual_review_only=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
    save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='complete'));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
