"""Original image bytes, frozen heldout local-pose experiment; separate from old gates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from core import image_binding,sha256
from heldout_anchor_pose import localize,POLICY
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-suffix',default='')
    args=parser.parse_args()
    if args.output_suffix not in ('','_v2'):raise ValueError('unsupported experiment suffix')
    source=ROOT/'artifacts/mendeley_visible_fan_scope_20261005'
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    registration_path=source/'registration/report.json'
    old=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    registration=json.loads(registration_path.read_text(encoding='utf-8'))['registration']
    if not registration['alignment_quality']['reliable']:raise ValueError('existing global gates rejected')
    matrix=np.array(registration['source_to_reference_homography'])
    output=ROOT/('artifacts/mendeley_heldout_anchor_pose_20261005'+args.output_suffix)
    if output.exists():raise FileExistsError('preserve prior experiment')
    files=[Path(__file__),Path(__file__).with_name('heldout_anchor_pose.py'),Path(__file__).with_name('local_anchor_pose.py'),
        Path(__file__).with_name('prepare_mendeley_scope.py'),scope_path,registration_path]
    inputs=[]
    for side in ['reference','inspection']:
        entry=old['original_sources'][side];path=Path(entry['path']);binding=image_binding(path)
        if any(binding[k]!=entry[k] for k in binding):raise ValueError('source drift')
        files.append(path);inputs.append(path)
    pins={str(p):sha256(p) for p in files};mainline=source_pins()
    output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'policy':POLICY,'pins':pins,'mainline_pins':mainline,
        'split_before_fit':True,'original_image_decode_required':True,'posthoc_development_only':True,
        'unchanged_old_final_gates':True,'GT_read':False,
        'stop_if':'input/code drift or any fit/check failure; no retry or threshold sweep'})
    started=time.perf_counter();images=[]
    for path in inputs:
        with Image.open(path) as opened:images.append(np.asarray(opened.convert('RGB')))
    results=[localize(*images,anchor,matrix) for anchor in scope['anchors']]
    if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=mainline:raise ValueError('source drift during experiment')
    report={'status':'complete','seconds':time.perf_counter()-started,'anchors':results,'source_pins':pins,
        'mainline_unchanged':True,'deployed':False,'reference_review_confirmed':False,
        'new_confirmed_connections':0,'decision':'insufficient_evidence','old_pose_gates_not_changed':True,
        'new_pose_proposals_not_connection_verdicts':True,'GT_read':False,'protocol_sha256':sha256(output/'protocol.json')}
    save(output/'report.json',report)
    print(json.dumps({'status':'complete','seconds':report['seconds'],'anchors':[{
        k:r.get(k) for k in ['id','localization_proposal_supported','training_count','heldout_count','training_inliers',
            'heldout_median_error_px','heldout_support_ratio','local_global_corner_disagreement_px','gates','reason']}
        for r in results]},ensure_ascii=False))


if __name__=='__main__':main()
