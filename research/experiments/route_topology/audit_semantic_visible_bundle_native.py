"""Independent flood-fill and scalar projection of unchanged native SAM pixels.

No shared bundle/component evaluator or mask repair; appearance inference not rerun.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_prompt_contrast import save,verify
from run_audit import source_pins
ROOT=Path(__file__).resolve().parents[2]

def flood_components(raw):
    h,w=raw.shape;remaining=set(np.flatnonzero(raw).tolist());parts=[]
    while remaining:
        seed=min(remaining);remaining.remove(seed);stack=[seed];part=[]
        while stack:
            p=stack.pop();part.append(p);y,x=divmod(p,w)
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if not (dx or dy) or not (0<=x+dx<w and 0<=y+dy<h):continue
                    q=(y+dy)*w+x+dx
                    if q in remaining:remaining.remove(q);stack.append(q)
        parts.append(part)
    return parts

def main():
    evidence=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    rp=ROOT/'artifacts/mendeley_semantic_visible_bundle_review_20261006/report.json'
    report=json.loads(rp.read_text(encoding='utf-8'))
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    prepared=json.loads((evidence/'preparation_report.json').read_text(encoding='utf-8'))['cases']
    assert report['status']=='complete';verify(protocol['pins'])
    assert source_pins()==protocol['mainline_pins']
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    observed={'reference':report['reference_observation'],**{c['id']:c['observation'] for c in report['cases']}}
    checks=[];total_components=0;total_masks=0
    for case in prepared:
        original=case['original_source'];assert sha256(original['path'])==original['image_sha256']
        observation=observed[case['id']];assert observation['source_binding']['image_sha256']==original['image_sha256']
        context=case['crop_context'];eligible=0
        if case['sam_inference_requested']:
            actual=np.asarray(Image.open(original['path']).convert('RGB').crop(context['crop_box_xyxy']))
            assert np.array_equal(actual,np.asarray(Image.open(context['source']['path']).convert('RGB')))
            audit={r['record_id']:r for r in observation['mask_audit']}
            for recipe in protocol['recipes']:
                run=evidence/case['id']/recipe
                manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
                for filename,d in manifest['verified_files'].items():assert sha256(filename)==d
                if recipe!='cable_plus_reference_anatomy_box':continue
                raw_report=json.loads((run/'sam/report.json').read_text(encoding='utf-8'))
                for index,score in enumerate(raw_report['scores'],1):
                    record_id=f'mask_{index:03d}';old=audit[record_id]
                    raw=np.asarray(Image.open(run/'sam'/(record_id+'.png')).convert('L'))>0
                    parts=flood_components(raw);total_components+=len(parts);total_masks+=1
                    ys,xs=np.where(raw);h,w=raw.shape
                    truncated=bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=w-2 or ys.max()>=h-2))
                    assert truncated==old['boundary_truncated'] and len(parts)==len(old['components'])
                    signatures=[]
                    for part in parts:
                        hits={a['id']:0 for a in scope['anchors']}
                        for anchor in scope['anchors']:
                            pose=next(p for p in case['anchors'] if p['id']==anchor['id'])
                            if not pose['localization_proposal_supported'] or not all(pose['gates'].values()):continue
                            matrix=np.asarray(pose['inspection_to_reference_local']);l,t,r,b=anchor['bbox_xyxy']
                            for flat in part:
                                y,x=divmod(flat,w);x+=context['crop_box_xyxy'][0];y+=context['crop_box_xyxy'][1]
                                qx=matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]
                                qy=matrix[1,0]*x+matrix[1,1]*y+matrix[1,2]
                                qw=matrix[2,0]*x+matrix[2,1]*y+matrix[2,2]
                                assert abs(qw)>1e-9
                                if l<=qx/qw<=r and t<=qy/qw<=b:hits[anchor['id']]+=1
                        signatures.append((len(part),tuple(sorted(hits.items()))))
                        if score>=.75 and not truncated and all(v>0 for v in hits.values()):eligible+=1
                    expected=[(c['pixel_count'],tuple(sorted(c['anchor_pixel_support'].items()))) for c in old['components']]
                    assert sorted(signatures)==sorted(expected)
        assert eligible==len(observation['eligible_native_components'])
        checks.append(dict(id=case['id'],native_two_anchor_components=eligible,
            inventory_verified=case['sam_inference_requested']))
    assert checks[0]['native_two_anchor_components']==1
    for check,case in zip(checks[1:],report['cases']):
        expected='same_visible_bundle_attachment_supported' if check['inventory_verified'] and check['native_two_anchor_components']==1 else 'insufficient_evidence'
        assert expected==case['comparison']['decision']
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    out=ROOT/'artifacts/mendeley_semantic_visible_bundle_native_audit_20261006';out.mkdir(exist_ok=False)
    save(out/'report.json',dict(status='PASS',source_report_sha256=sha256(rp),cases=checks,
        native_masks_rechecked=total_masks,native_components_rechecked=total_components,
        independent_python_flood_fill_and_scalar_projection=True,
        source_crops_and_manifest_bytes_verified=True,shared_appearance_evidence_not_rerun=True,
        production_unchanged=True,electrical_connections_confirmed=0,deployed=False))
    print(json.dumps(checks))
if __name__=='__main__':main()
