"""One preregistered source trial. No SAM, demo fitting, or gate relaxation."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from component_cues import fit_head, prediction, probability, gate
from socket_lbp_candidate import descriptor, resolve, POLICY
from run_audit import source_pins
from run_review import save

ROOT = Path(__file__).resolve().parents[2]

def main():
    lp = ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    pp = ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json'
    bp = ROOT/'artifacts/mendeley_semantic_visible_pose_rescue_20261006/report.json'
    plan = ROOT/'artifacts/SOCKET_TEXTURE_PREREGISTRATION_20261008.md'
    output = ROOT/'artifacts/mendeley_socket_lbp_source_20261008'
    before = source_pins()
    partition = json.loads(pp.read_text(encoding='utf-8'))
    labels = json.loads(lp.read_text(encoding='utf-8'))
    base = json.loads(bp.read_text(encoding='utf-8'))
    if not base['strict_net_source_gain']: raise ValueError('qualified baseline required')
    by_id = {r['id']:r for r in labels['rows']}
    fitting = [by_id[i] for i in partition['fit_ids']]
    calibration = [by_id[i] for i in partition['calibration_ids']]
    previous = {c['id']:c for c in base['cases']}
    if set(previous) != set(partition['calibration_ids']): raise ValueError('source split differs')
    files = [lp, pp, bp, plan, Path(__file__), Path(__file__).with_name('socket_lbp_candidate.py'),
             Path(__file__).with_name('component_cues.py')]
    pins = {str(p):sha256(p) for p in files}
    output.mkdir(exist_ok=False)
    save(output/'protocol.json', dict(policy=POLICY, pins=pins, mainline_pins=before,
        fit_ids=partition['fit_ids'], calibration_ids=partition['calibration_ids'],
        assistant_qualitative_labels_not_human_GT=True, demo_images_read=False,
        repeated_source_development_not_blind_validation=True,
        acceptance='zero wrong, zero old loss, strict gain; nine unanimous unique labels',
        head=dict(steps=1000, learning_rate=.05, l2=.02, std_floor=.01)))
    start = time.monotonic()
    features = {}; probes = {}; records = []
    for row in fitting + calibration:
        if sha256(row['source_path']) != row['source_sha256']: raise ValueError('source image drift')
        if not row['pose']['localization_proposal_supported']: raise ValueError('source pose unsupported')
        rgb = np.asarray(Image.open(row['source_path']).convert('RGB'))
        pose = np.asarray(row['pose']['inspection_to_reference_local'])
        offsets = [(x,y) for x in [-2,0,2] for y in [-2,0,2]] if row['id'] in previous else [(0,0)]
        batch = []
        for x,y in offsets:
            h = np.array([[1,0,-1600+x],[0,1,-1000+y],[0,0,1]]) @ pose
            if not cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),h,(100,50),flags=cv2.INTER_NEAREST).all():
                raise ValueError('patch outside original source')
            patch = cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
            f = descriptor(patch); batch.append(f)
            if (x,y) == (0,0):
                if sha256(row['patch_path']) != row['patch_sha256']: raise ValueError('stored patch drift')
                if not np.array_equal(patch,np.asarray(Image.open(row['patch_path']).convert('RGB'))):
                    raise ValueError('original patch differs')
                features[row['id']] = f
        probes[row['id']] = batch
        records.append(dict(id=row['id'], feature=features[row['id']].tolist(),
                            probe_features=[f.tolist() for f in batch], source_sha256=row['source_sha256']))
        save(output/'progress.json', dict(status='extracting', completed=len(records), total=197))
    model = fit_head([features[r['id']] for r in fitting], [r['visual_label'] for r in fitting])
    cal = {k:[dict(id=r['id'],score=probability(model,features[r['id']]) if k == 0 else
                   1-probability(model,features[r['id']])) for r in calibration if r['visual_label']==k] for k in [0,1]}
    rows=[]; gains=[]; losses=[]
    for row in calibration:
        old = previous[row['id']]['prediction']['visual_label_candidate']
        ps = [prediction(model,f,cal,excluded_id=row['id']) for f in probes[row['id']]]
        final = resolve(old,[p['visual_label_candidate'] for p in ps])
        if old is None and final == row['visual_label']: gains.append(row['id'])
        if old is not None and final != old: losses.append(row['id'])
        rows.append(dict(id=row['id'], visual_label=row['visual_label'], old_candidate=old,
                         probes=ps,prediction=dict(visual_label_candidate=final)))
    score=gate(rows); passed=score['passed'] and not losses and bool(gains)
    if source_pins()!=before or any(sha256(p)!=digest for p,digest in pins.items()): raise ValueError('input/code/E drift')
    save(output/'features.json',dict(records=records))
    save(output/'model.json',{**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()},
                              'calibration':cal})
    result=dict(status='complete',seconds=time.monotonic()-start,source_gate=score,
        strict_net_source_gain=passed,new_source_support=gains,old_supported_losses=losses,cases=rows,
        automatic_topology_new_hits=0,electrical_connections_confirmed=0,
        deployed=False, demo_inference_run=False, next_action='independent source replay' if passed else 'stop; no demo or SAM')
    save(output/'report.json',result); save(output/'progress.json',dict(status='complete',strict_net_source_gain=passed))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))

if __name__=='__main__': main()
