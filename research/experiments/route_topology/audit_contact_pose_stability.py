"""Fixed source-wide +/-2px pose perturbation, no threshold fitting or deployment."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_positive_contact_gate import bright_contacts

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_contact_pose_stability_20261006';out.mkdir(exist_ok=False)
    labels_path=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    split_path=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json'
    report_path=ROOT/'artifacts/mendeley_positive_contact_source_20261006/report.json'
    mask_path=report_path.parent/'fit_contact_support.png'
    labels={r['id']:r for r in json.loads(labels_path.read_text(encoding='utf-8'))['rows']}
    split=json.loads(split_path.read_text(encoding='utf-8'))
    report=json.loads(report_path.read_text(encoding='utf-8'))
    mask=np.asarray(Image.open(mask_path))>0;threshold=report['fit_threshold']
    before=source_pins();pins={str(p):sha256(p) for p in [Path(__file__),labels_path,split_path,report_path,mask_path,Path(__file__).with_name('run_positive_contact_gate.py')]}
    offsets=[(x,y) for x in [-2,0,2] for y in [-2,0,2]]
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,mainline_pins=before,
        offsets=offsets,mask_and_threshold_frozen=True,all197_original_sources=True,
        reviewed_poses_reused=True,perturbations_are_diagnostics_not_independent_observers=True,
        no_parameters_updated=True,no_inspection_GT_or_images_read=True),indent=2),encoding='utf-8')
    rows=[]
    for i in split['fit_ids']+split['calibration_ids']:
        r=labels[i]
        if sha256(r['source_path'])!=r['source_sha256']:raise ValueError('source drift')
        rgb=np.asarray(Image.open(r['source_path']).convert('RGB'))
        pose=np.asarray(r['pose']['inspection_to_reference_local']);scores=[]
        for dx,dy in offsets:
            matrix=np.array([[1,0,-1600+dx],[0,1,-1000+dy],[0,0,1]])@pose
            patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
            scores.append(float(bright_contacts(patch)[mask].mean()))
        nominal=scores[offsets.index((0,0))]
        rows.append(dict(id=i,visual_label=r['visual_label'],scores=scores,nominal_score=nominal,
            nominal_positive=nominal>threshold,all_offsets_positive=min(scores)>threshold,
            all_offsets_negative=max(scores)<=threshold,stable=(min(scores)>threshold or max(scores)<=threshold)))
    byid={r['id']:r for r in rows}
    gains={i:byid[i] for i in report['source_gains']}
    result=dict(status='complete',source_count=len(rows),cases=rows,nominal_source_gains_stability=gains,
        calibration_cue_flips=[i for i in split['calibration_ids'] if not byid[i]['stable']],
        source_strict_gains_all_offsets_positive=all(r['all_offsets_positive'] for r in gains.values()),
        not_a_new_classifier_gate=True,mainline_unchanged=True,deployed=False,new_confirmed_connections=0)
    if before!=source_pins() or any(sha256(p)!=d for p,d in pins.items()):raise ValueError('E/code/input drift')
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(source_count=len(rows),gain_scores={i:r['scores'] for i,r in gains.items()},
        gains_stable=result['source_strict_gains_all_offsets_positive'],calibration_cue_flip_count=len(result['calibration_cue_flips']))))
if __name__=='__main__':main()
