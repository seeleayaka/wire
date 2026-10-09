"""Diagnostic only: native bright-site overlap is not socket occupancy proof."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from core import sha256
from run_prompt_contrast import save, verify
from run_audit import source_pins
from run_positive_contact_gate import bright_contacts
from run_paired_evidence import load_run
from run_review import verified_run

ROOT = Path(__file__).resolve().parents[2]

def native_sites(rgb, matrix, support, crop):
    """Round to unchanged source pixels, deduplicate sites, never modify masks."""
    h, w = support.shape
    patch = cv2.warpPerspective(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR)
    ys, xs = np.where(support & bright_contacts(patch))
    sites = np.column_stack((xs, ys, np.ones(len(xs)))) @ np.linalg.inv(matrix).T
    if len(sites) and np.any(np.abs(sites[:, 2]) < 1e-9): raise ValueError('projection horizon')
    points = np.rint(sites[:, :2] / sites[:, 2:]).astype(int)
    result = set()
    for x, y in points:
        if not (0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]): raise ValueError('outside source')
        cx, cy = x-crop[0], y-crop[1]
        if not (0 <= cx < crop[2]-crop[0] and 0 <= cy < crop[3]-crop[1]): raise ValueError('outside crop')
        if bright_contacts(rgb[y:y+1, x:x+1])[0,0]: result.add((int(cx), int(cy)))
    return sorted(result)

def main():
    src = ROOT / 'artifacts/mendeley_typed_affine_bundle_20261006'
    protocol = json.loads((src/'protocol.json').read_text(encoding='utf-8'))
    prepared = json.loads((src/'preparation_report.json').read_text(encoding='utf-8'))['cases']
    scope = json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    anchor = next(a for a in scope['anchors'] if a['kind']=='wire_entry_socket')
    support_path = ROOT / 'artifacts/mendeley_positive_contact_source_20261006/fit_contact_support.png'
    support = np.asarray(Image.open(support_path)) > 0
    l,t,r,b = anchor['bbox_xyxy']; assert support.shape == (b-t,r-l)
    pins = {str(support_path):sha256(support_path), str(src/'protocol.json'):sha256(src/'protocol.json'), str(Path(__file__)):sha256(__file__)}
    verify(protocol['pins']); assert source_pins() == protocol['mainline_pins']
    rows = []; overlays = []
    for case in prepared:
        origin = case['original_source']; assert sha256(origin['path']) == origin['image_sha256']
        rgb = np.asarray(Image.open(origin['path']).convert('RGB'))
        pose = next(p for p in case['anchors'] if p['id']==anchor['id'])
        assert pose['localization_proposal_supported'] and all(pose['gates'].values())
        matrix = np.array([[1,0,-l],[0,1,-t],[0,0,1]]) @ np.asarray(pose['inspection_to_reference_local'])
        crop = case['crop_context']['crop_box_xyxy']
        sites = native_sites(rgb,matrix,support,crop); masks = []
        base = Image.fromarray(rgb).crop(crop); drawer = ImageDraw.Draw(base)
        for x,y in sites: drawer.ellipse((x-2,y-2,x+2,y+2),outline='#00aaee',width=1)
        for recipe in protocol['recipes']:
            run = src/case['id']/recipe; origin_record, records = load_run(run)
            verified_run(run,origin_record['image_path'])
            for record in records:
                raw = np.asarray(Image.open(record['source_mask_path']).convert('L')) > 0
                assert raw.shape == (crop[3]-crop[1],crop[2]-crop[0])
                hits = [list(p) for p in sites if raw[p[1],p[0]]]
                masks.append(dict(recipe=recipe,record_id=record['record_id'],score=record['score'],
                    distinct_native_bright_sites_overlapped=len(hits),native_xy=hits,
                    mask_sha256=sha256(record['source_mask_path'])))
                if record['score']>=.75 and recipe.endswith('_box'):
                    for x,y in hits: drawer.ellipse((x-4,y-4,x+4,y+4),outline='#ee4400',width=2)
        overlays.append((case['id'],base))
        rows.append(dict(id=case['id'],phenotype=case['phenotype'],distinct_native_sites=len(sites),
            native_sites=sites,masks=masks))
    verify(protocol['pins']); verify(pins); assert source_pins()==protocol['mainline_pins']
    out = ROOT/'artifacts/mendeley_typed_affine_contact_sites_20261006'; out.mkdir(exist_ok=False)
    save(out/'report.json',dict(status='complete',pins=pins,cases=rows,decision_gate_unchanged=True,
        sites_not_certified_metal_contacts=True,site_absence_not_evidence_of_disconnection=True,
        masks_unchanged=True,production_unchanged=True,deployed=False))
    for identity,base in overlays: base.save(out/(identity+'_sites.png'))
    print(json.dumps([dict(id=row['id'],sites=row['distinct_native_sites'],high_score_box_overlaps=[m['distinct_native_bright_sites_overlapped'] for m in row['masks'] if m['score']>=.75 and m['recipe'].endswith('_box')]) for row in rows]))

if __name__=='__main__': main()
