"""Separate broad-box contact from sampled bright-contact evidence, no override."""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_paired_evidence import load_run
from run_review import verified_run
from run_prompt_contrast import verify
from run_positive_contact_gate import bright_contacts

ROOT=Path(__file__).resolve().parents[2]
def main():
    src=ROOT/'artifacts/mendeley_compound_bundle_pair_20261006'
    protocol=json.loads((src/'protocol.json').read_text(encoding='utf-8'))
    preparation=json.loads((src/'preparation_report.json').read_text(encoding='utf-8'))
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    support_path=ROOT/'artifacts/mendeley_positive_contact_source_20261006/fit_contact_support.png'
    support=np.asarray(Image.open(support_path))>0
    out=ROOT/'artifacts/mendeley_native_contact_conflict_audit_20261006';out.mkdir(exist_ok=False)
    rows=[]
    for row in preparation['cases']:
        rgb=np.asarray(Image.open(row['original_source']['path']).convert('RGB'))
        pose=next(a for a in row['anchors'] if a['id']=='FAN_CPU')
        h=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.asarray(pose['inspection_to_reference_local'])
        probe=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
        positive=support&bright_contacts(probe)
        ys,xs=np.where(positive);q=np.column_stack((xs,ys,np.ones(len(xs))))@np.linalg.inv(h).T
        q=np.rint(q[:,:2]/q[:,2:]).astype(int)
        crop=row['crop_context']['crop_box_xyxy'];local=q-np.asarray(crop[:2])
        actual_native_positive=[]
        for original_xy,local_xy in zip(q,local):
            x,y=original_xy;cx,cy=local_xy
            if not (0<=x<rgb.shape[1] and 0<=y<rgb.shape[0] and 0<=cx<crop[2]-crop[0] and 0<=cy<crop[3]-crop[1]):raise ValueError('probe outside original crop')
            if bright_contacts(rgb[y:y+1,x:x+1])[0,0]:actual_native_positive.append([int(cx),int(cy)])
        masks=[]
        for recipe in protocol['recipes']:
            run=src/row['id']/recipe;origin,records=load_run(run);verified_run(run,origin['image_path'])
            for r in records:
                raw=np.asarray(Image.open(r['source_mask_path']).convert('L'))>0
                if raw.shape!=(crop[3]-crop[1],crop[2]-crop[0]):raise ValueError('mask frame mismatch')
                native_hit=[p for p in actual_native_positive if raw[p[1],p[0]]]
                masks.append(dict(recipe=recipe,record_id=r['record_id'],score=r['score'],
                    native_positive_probe_hits=len(native_hit),native_positive_probe_xy=native_hit,
                    source_mask_path=r['source_mask_path'],source_mask_sha256=sha256(r['source_mask_path'])))
        rows.append(dict(id=row['id'],nominal_warped_contact_sites=int(positive.sum()),
            actual_native_positive_probe_count=len(actual_native_positive),native_positive_probe_xy=actual_native_positive,
            masks=masks,probe_points_not_certified_metal_contacts=True,
            original_socket_evidence=row['socket_evidence']))
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    report=dict(status='complete',cases=rows,source_support_sha256=sha256(support_path),
        conflict_gate_unchanged=True,does_not_establish_physical_disconnection=True,
        mask_pixels_changed=False,mainline_unchanged=True,deployed=False,
        comparison_still='insufficient_evidence',inherited_reason='exposed_socket_and_high_score_mask_conflict')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([dict(id=r['id'],native_probes=r['actual_native_positive_probe_count'],high_score_box_recipe_contacts=[dict(id=m['record_id'],hits=m['native_positive_probe_hits']) for m in r['masks'] if m['score']>=.75 and m['recipe'].endswith('_box')]) for r in rows]))
if __name__=='__main__':main()
