"""Fresh local OCR on fixed existing originals; no remote image/API/GT reads."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core import image_binding, sha256
from identity_ocr import inverse_rotation, summarize_readings


ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = ROOT / 'artifacts/route_identity_dependencies_20261005/packages'


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/route_identity_ocr_20261005')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(DEPENDENCIES))
    from rapidocr_onnxruntime import RapidOCR
    project = Path('E:/PythonProject10')
    data = project / 'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
    cases = [(f'cabinet_{i}', Path(f'C:/Users/HUAWEI/Desktop/案例/线材柜子/{i}/{i}.png')) for i in [1,2,5]]
    cases += [('pc_normal_reference', data / 'train01/normal_073.JPG'), ('pc_route_change', data / 'test01/misrouted_001.JPG')]
    # Fixed before inference; missing sources are errors, not quietly substituted.
    sources = {name:image_binding(path) for name,path in cases}
    models = {str(p):sha256(p) for p in sorted((DEPENDENCIES / 'rapidocr_onnxruntime/models').glob('*.onnx'))}
    if len(models) != 3:
        raise ValueError('expected exactly 3 bundled OCR model files')
    protocol = {'created_at':datetime.now(timezone.utc).isoformat(), 'sources':sources,
        'models':models, 'package':'rapidocr_onnxruntime==1.4.4', 'scale':2,
        'quarter_turns':[0,1,2,3], 'threads':2, 'default_OCR_thresholds':True,
        'nomination_high_score':.9, 'spatial_group_pairwise_iou':.5,
        'same_weights_view_observers':1, 'no_automatic_port_or_wire_confirmation':True,
        'no_model_or_crop_tuning':True, 'source_gt_read':False, 'new_local_OCR':True,
        'publications':{'package':'https://pypi.org/project/rapidocr-onnxruntime/1.4.4/',
                        'wheel_sha256':'971d7d5f223a7a808662229df1ef69893809d8457d834e6373d3854bc1782cbf'}}
    save(out / 'protocol.json', protocol)
    save(out / 'progress.json', {'status':'building_OCR', 'completed':0, 'total':len(cases)})
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    summaries = []
    started = time.perf_counter()
    for name,path in cases:
        target = out / name
        target.mkdir()
        with Image.open(path) as opened:
            original = opened.convert('RGB')
            doubled = np.asarray(original.resize((original.width*2, original.height*2), Image.Resampling.LANCZOS))
        rows = []
        for rotation in [0,1,2,3]:
            save(out / 'progress.json', {'status':'OCR', 'case':name, 'rotation':rotation, 'completed':len(summaries), 'total':len(cases)})
            # API ndarray convention is BGR, matching its local image reader.
            view = np.ascontiguousarray(np.rot90(doubled, rotation)[:, :, ::-1])
            result, elapsed = engine(view)
            save(target / f'view_{rotation}_raw.json', {'rows':result, 'elapsed':elapsed})
            for index, (polygon, text, score) in enumerate(result or []):
                mapped = inverse_rotation(polygon, doubled.shape[1], doubled.shape[0], rotation, 2)
                points = np.asarray(mapped)
                points[:,0] = np.clip(points[:,0], 0, original.width)
                points[:,1] = np.clip(points[:,1], 0, original.height)
                box = [float(points[:,0].min()), float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError('OCR mapped a nonpositive box')
                rows.append({'record_id':f'v{rotation}_{index:03}', 'text':text, 'score':float(score),
                             'bbox_xyxy':box, 'polygon_source_xy':points.tolist(), 'view_quarter_turns':rotation})
        nomination = summarize_readings(rows)
        nomination.update(image_binding=sources[name], records=rows, fresh_local_OCR=True,
                          topology_decision='insufficient_evidence')
        save(target / 'report.json', nomination)
        canvas = original.resize((original.width*2,original.height*2), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(canvas)
        for g in nomination['groups']:
            box = [v*2 for v in g['bbox_xyxy']]
            color = '#00877b' if g['ocr_high_score_consistent'] else '#bc751f'
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], max(0,box[1]-18)), g['group_id']+':'+g['best_text'], font=font, fill=color, stroke_width=1, stroke_fill='white')
        canvas.save(target / 'text_overlay.png')
        summaries.append({'case':name, 'raw_readings':len(rows), 'spatial_groups':len(nomination['groups']),
                          'high_score_consistent':sum(g['ocr_high_score_consistent'] for g in nomination['groups']),
                          'repeated_text_groups':nomination['repeated_text_groups'], 'confirmed_ports':0,
                          'topology_decision':'insufficient_evidence'})
        if image_binding(path) != sources[name]:
            raise ValueError('source image changed')
    if {p:sha256(p) for p in models} != models:
        raise ValueError('OCR weights changed')
    save(out / 'report.json', {'status':'complete', 'seconds':time.perf_counter()-started, 'cases':summaries,
         'verified_sources_and_models_unchanged':True, 'new_confirmed_connections':0,
         'new_electrical_accuracy':None, 'deployed':False, 'protocol_sha256':sha256(out/'protocol.json')})
    save(out / 'progress.json', {'status':'complete', 'completed':len(cases), 'total':len(cases)})
    print(json.dumps({'status':'complete', 'cases':summaries},ensure_ascii=False))


if __name__ == '__main__':
    main()
