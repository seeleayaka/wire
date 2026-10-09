"""Fresh decode and pose from fixed source inventory; no cached image/features."""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from core import sha256
from run_audit import source_pins
from heldout_anchor_pose import localize
from socket_edge_caliper import compare, POLICY

ROOT=Path(__file__).resolve().parents[2]

def main():
    start=time.monotonic()
    scope=json.loads((ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json').read_text(encoding='utf-8'))
    approval=json.loads((ROOT/'artifacts/mendeley_reference_confirmed_20261006/approval_record.json').read_text(encoding='utf-8'))
    before=source_pins()
    if before!=approval['mainline_pins']: raise ValueError('mainline drift')
    inventory=json.loads((ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json').read_text(encoding='utf-8'))['cases']
    out=ROOT/'artifacts/mendeley_socket_edge_caliper_20261006';out.mkdir(exist_ok=False)
    code={str(Path(__file__)):sha256(Path(__file__)),str(Path(__file__).with_name('socket_edge_caliper.py')):sha256(Path(__file__).with_name('socket_edge_caliper.py'))}
    (out/'protocol.json').write_text(json.dumps(dict(policy=POLICY,code_pins=code,mainline_pins=before,selection='same four SHA-first FIT controls plus approved reference',source_labels_not_used_to_fit=True,not_insertion_depth=True),indent=2),encoding='utf-8')
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    reference_patch=reference[1000:1050,1600:1700]
    rows=[]; patches=[]
    for row in inventory:
        source=row['original_source']; path=Path(source['path'])
        if sha256(path)!=source['image_sha256']: raise ValueError('original drift')
        rgb=np.asarray(Image.open(path).convert('RGB'))
        result=dict(id=row['id'],source=source,decision='insufficient_evidence')
        if row['id']=='reference': matrix=np.eye(3); pose=dict(localization_proposal_supported=True)
        else:
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            pose=localize(reference,rgb,anchor,np.array(registration['source_to_reference_homography'])) if registration['alignment_quality']['reliable'] else dict(localization_proposal_supported=False)
            matrix=np.array(pose['inspection_to_reference_local']) if pose['localization_proposal_supported'] else None
        result['fresh_pose']=pose
        if matrix is not None:
            transform=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@matrix
            valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),transform,(100,50),flags=cv2.INTER_NEAREST)
            if not valid.all(): raise ValueError('out of original pixels')
            patch=cv2.warpPerspective(rgb,transform,(100,50),flags=cv2.INTER_LINEAR)
            Image.fromarray(patch).save(out/(row['id']+'.png')); patches.append((row['id'],patch))
            result.update(compare(reference_patch,patch))
        rows.append(result)
    unchanged=before==source_pins() and all(sha256(p)==d for p,d in code.items())
    if not unchanged: raise ValueError('source/code drift')
    # Qualitative fixed-source gate is reported only AFTER all predictions.
    expected={r['id']:r['source_control_annotation']['visual_label']==1 for r in inventory if r['source_control_annotation']}
    passed=all(r.get('reference_geometry_compatible',False)==expected.get(r['id'],True) for r in rows)
    report=dict(status='complete',cases=rows,seconds=time.monotonic()-start,source_gate_passed=passed,
                gate_meaning='geometry compatibility agrees with four assistant qualitative phenotypes; NOT electrical GT',
                mainline_unchanged=unchanged,deployed=False,new_confirmed_connections=0,
                source_only=True,inspection_results_not_read=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(620, len(patches)*130),'white'); d=ImageDraw.Draw(sheet)
    for i,(identity,patch) in enumerate(patches):
        sheet.paste(Image.fromarray(patch).resize((400,100)),(0,i*130+25)); d.text((5,i*130+5),identity,fill='black')
    sheet.save(out/'source_contact_sheet.png')
    print(json.dumps(dict(report=str(out/'report.json'),source_gate_passed=passed,seconds=report['seconds'],scores={r['id']:r.get('bilateral_support') for r in rows})))

if __name__=='__main__': main()
