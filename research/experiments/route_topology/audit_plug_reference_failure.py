"""Diagnose unchanged native plug mask, not a new inference or gate revision."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from run_prompt_contrast import digest, save
from run_audit import source_pins

ROOT = Path(__file__).resolve().parents[2]

def main():
    source = ROOT / 'artifacts/mendeley_plug_source_probe_v2_20261006'
    out = ROOT / 'artifacts/mendeley_plug_reference_failure_audit_20261006'
    out.mkdir(exist_ok=False)
    pins = source_pins()
    protocol = json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    case = protocol['cases'][0]
    crop_path = Path(case['source']['path'])
    original_path = Path(case['original_source']['path'])
    mask_path = source/'reference/plug_plus_socket_anatomy_box/sam/mask_001.png'
    assert digest(crop_path) == case['source']['image_sha256']
    assert digest(original_path) == case['original_source']['image_sha256']
    crop = np.asarray(Image.open(crop_path).convert('RGB'))
    fresh_crop = np.asarray(Image.open(original_path).convert('RGB').crop(case['crop_box_xyxy']))
    assert np.array_equal(crop, fresh_crop)
    mask = np.asarray(Image.open(mask_path).convert('L')) > 0
    assert mask.shape == crop.shape[:2]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = [dict(label=i, area=int(stats[i,4]), bbox_xywh=stats[i,:4].tolist(),
                       centroid=centroids[i].tolist()) for i in range(1,count)]
    ys,xs = np.where(mask)
    truncated = bool(xs.min()<=1 or ys.min()<=1 or xs.max()>=mask.shape[1]-2 or ys.max()>=mask.shape[0]-2)
    overlay = crop.copy()
    overlay[mask] = (.55*crop[mask] + .45*np.array([255,40,70])).astype(np.uint8)
    sheet = Image.new('RGB', (800,340), 'white')
    sheet.paste(Image.fromarray(crop).resize((400,300)), (0,35))
    sheet.paste(Image.fromarray(overlay).resize((400,300)), (400,35))
    draw = ImageDraw.Draw(sheet)
    draw.text((5,10), 'Fresh original crop (no mask edits)', fill='black')
    draw.text((405,10), 'Unchanged SAM mask overlay', fill='black')
    sheet.save(out/'reference_overlay.png')
    assert source_pins() == pins
    raw_report=json.loads((mask_path.parent/'report.json').read_text(encoding='utf-8'))
    save(out/'report.json',dict(status='complete', components=components,
         native_component_count=count-1, mask_pixels=int(mask.sum()), boundary_truncated=truncated,
         pixel_identity_to_original_crop=True, mask_sha256=digest(mask_path),
         model_score=raw_report['scores'][0], text_only_instances=0, box_assisted_instances=raw_report['instance_count'],
         failed_checks=['single_native_component'] if count-1!=1 else [],
         no_mask_pixels_changed=True, new_model_inference=False,
         semantic_identity_verified=False, electrical_connection_verified=False,
         production_pins=pins, production_unchanged=True, deployed=False))
    print(json.dumps(components))

if __name__=='__main__':main()
