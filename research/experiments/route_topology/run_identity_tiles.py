"""One fixed local-OCR resolution experiment; no per-image ROI tuning.

All five original sources get the same full view + nine half-size overlapping
tiles, four rotations, same OCR weights/default thresholds, one observer.
Crop-boundary text is rejected. No authenticated terminal IDs are emitted.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core import image_binding, sha256
from identity_ocr import inverse_rotation, summarize_readings
from run_identity_ocr import ROOT, DEPENDENCIES, save


def fixed_regions(width,height):
    yield 'full',[0,0,width,height]
    # Generic 50% extent, 25% stride. Rounding covers odd image dimensions.
    for row in range(3):
        for col in range(3):
            yield f'tile_{row}_{col}',[round(col*width/4),round(row*height/4),
                                     round((col+2)*width/4),round((row+2)*height/4)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/route_identity_ocr_tiles_20261005')
    out=parser.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    project=Path('E:/PythonProject10')
    data=project/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
    cases=[(f'cabinet_{i}',Path(f'C:/Users/HUAWEI/Desktop/案例/线材柜子/{i}/{i}.png')) for i in [1,2,5]]
    cases += [('pc_normal_reference',data/'train01/normal_073.JPG'),('pc_route_change',data/'test01/misrouted_001.JPG')]
    sources={name:{'path':str(path),**image_binding(path)} for name,path in cases}
    models={str(p):sha256(p) for p in sorted((DEPENDENCIES/'rapidocr_onnxruntime/models').glob('*.onnx'))}
    if len(models)!=3:
        raise ValueError('exactly 3 bundled models required')
    pins={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('identity_ocr.py'),Path(__file__).with_name('run_identity_ocr.py')]}
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),'sources':sources,'models':models,
        'source_pins':pins,'package':'rapidocr_onnxruntime==1.4.4','scale':2,
        'quarter_turns':[0,1,2,3],'regions':{name:list(fixed_regions(*s['image_size'])) for name,s in sources.items()},
        'crop_boundary_margin_original_px':2,'OCR_thresholds':'unchanged defaults',
        'nomination_high_score':.9,'pairwise_spatial_iou':.5,'one_model_observer':True,
        'no_GT_reads':True,'no_per_image_ROI_or_threshold_tuning':True,'confirmed_ports':0}
    save(out/'protocol.json',protocol)
    sys.path.insert(0,str(DEPENDENCIES))
    from rapidocr_onnxruntime import RapidOCR
    engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=2)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    started=time.perf_counter();summaries=[]
    for name,path in cases:
        target=out/name;target.mkdir()
        with Image.open(path) as opened:
            original=opened.convert('RGB')
        rows=[];rejected=[]
        for region_id,box in fixed_regions(*original.size):
            tile=original.crop(box)
            doubled=np.asarray(tile.resize((tile.width*2,tile.height*2),Image.Resampling.LANCZOS))
            for rotation in range(4):
                save(out/'progress.json',{'status':'OCR','case':name,'region':region_id,'rotation':rotation,'completed':len(summaries),'total':len(cases)})
                result,elapsed=engine(np.ascontiguousarray(np.rot90(doubled,rotation)[:,:,::-1]))
                save(target/f'{region_id}_v{rotation}_raw.json',{'rows':result,'elapsed':elapsed})
                for index,(polygon,text,score) in enumerate(result or []):
                    points=np.asarray(inverse_rotation(polygon,tile.width*2,tile.height*2,rotation,2))
                    identity=f'{region_id}_v{rotation}_{index:03}'
                    local=[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
                    if region_id!='full' and (local[0]<=2 or local[1]<=2 or local[2]>=tile.width-2 or local[3]>=tile.height-2):
                        rejected.append({'record_id':identity,'reason':'text_touches_crop_boundary','text':text,'score':float(score)})
                        continue
                    points += np.array(box[:2])
                    points[:,0]=np.clip(points[:,0],0,original.width)
                    points[:,1]=np.clip(points[:,1],0,original.height)
                    bbox=[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
                    rows.append({'record_id':identity,'text':text,'score':float(score),'bbox_xyxy':bbox,
                                 'polygon_source_xy':points.tolist(),'view_quarter_turns':rotation,'source_crop_xyxy':box})
        nomination=summarize_readings(rows)
        nomination.update(image_binding=image_binding(path),records=rows,crop_boundary_rejections=rejected,
                          fresh_local_OCR=True,topology_decision='insufficient_evidence')
        save(target/'report.json',nomination)
        canvas=original.resize((original.width*2,original.height*2),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(canvas)
        for g in nomination['groups']:
            bbox=[v*2 for v in g['bbox_xyxy']];color='#00877b' if g['ocr_high_score_consistent'] else '#bc751f'
            draw.rectangle(bbox,outline=color,width=2)
            draw.text((bbox[0],max(0,bbox[1]-18)),g['group_id']+':'+g['best_text'],font=font,fill=color,stroke_width=1,stroke_fill='white')
        canvas.save(target/'text_overlay.png')
        summaries.append({'case':name,'raw_readings_retained':len(rows),'crop_boundary_rejections':len(rejected),
                          'spatial_groups':len(nomination['groups']),
                          'high_score_consistent':sum(g['ocr_high_score_consistent'] for g in nomination['groups']),
                          'repeated_text_groups':nomination['repeated_text_groups'],'confirmed_ports':0,
                          'topology_decision':'insufficient_evidence'})
        if image_binding(path)!={k:sources[name][k] for k in ['image_sha256','image_size','coordinate_frame']}:
            raise ValueError('original source drift')
    if any(sha256(p)!=digest for p,digest in {**models,**pins}.items()):
        raise ValueError('weights/code drift')
    save(out/'report.json',{'status':'complete','seconds':time.perf_counter()-started,'cases':summaries,
         'fresh_local_OCR_calls':200,'verified_sources_weights_code_unchanged':True,
         'new_confirmed_connections':0,'new_electrical_accuracy':None,'deployed':False,
         'protocol_sha256':sha256(out/'protocol.json')})
    save(out/'progress.json',{'status':'complete','completed':len(cases),'total':len(cases)})
    print(json.dumps({'status':'complete','cases':summaries},ensure_ascii=False))


if __name__=='__main__':
    main()
