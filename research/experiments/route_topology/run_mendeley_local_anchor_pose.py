"""Fresh local SIFT from originals around frozen reference anchors; GT-free diagnostic."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from core import image_binding,sha256
from local_anchor_pose import localize_anchor
from run_review import save
from visible_lead_scope import validate_scope

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_visible_fan_scope_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    report_path=source/'registration/report.json'
    registration=json.loads(report_path.read_text(encoding='utf-8'))
    images=[];pins={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('local_anchor_pose.py'),scope_path,report_path]}
    for side in ['reference','inspection']:
        item=protocol['original_sources'][side];path=Path(item['path'])
        binding=image_binding(path)
        if any(binding[k]!=item[k] for k in binding):raise ValueError('original source drift')
        if side=='reference':validate_scope(scope,binding)
        pins[str(path)]=sha256(path)
        with Image.open(path) as opened:images.append(np.asarray(opened.convert('RGB')))
    if not registration['registration']['alignment_quality']['reliable']:raise ValueError('global registration required')
    output=ROOT/'artifacts/mendeley_local_anchor_pose_20261005'
    if output.exists():raise FileExistsError('fresh output required')
    start=time.perf_counter();matrix=np.array(registration['registration']['source_to_reference_homography'])
    results=[localize_anchor(*images,anchor,matrix) for anchor in scope['anchors']]
    if any(sha256(p)!=d for p,d in pins.items()):raise ValueError('source drift during local verification')
    output.mkdir(exist_ok=False)
    result={'status':'complete','seconds':time.perf_counter()-start,'anchors':results,'pins':pins,
        'inputs':'original image bytes, fresh local SIFT','reused_global_registration':True,
        'reference_review_confirmed':False,'GT_read':False,'new_confirmed_connections':0,
        'topology_decision':'insufficient_evidence','deployed':False}
    save(output/'report.json',result)
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],'anchors':[
        {k:r.get(k) for k in ['id','reliable_localization','mutual_static_context_matches','inliers','median_error_px',
           'local_global_corner_disagreement_px','reason']} for r in results]},ensure_ascii=False))


if __name__=='__main__':main()
