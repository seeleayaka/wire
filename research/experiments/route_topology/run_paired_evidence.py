"""Verified source-pixel SAM evidence -> unconfirmed cross-view proposals.

Uses the existing mainline registration gates unchanged. This is an optional
workspace research entry, not an electrical correctness/continuity detector.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

from core import sha256, image_binding, extract_mask
from run_review import verified_run, save, read
from paired_evidence import propose_correspondences, project

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('E:/PythonProject10')


def lift_verified_context(origin, records, context_protocol):
    """Use full-frame registration context without stitching missing pixels.

    Every crop must be a byte-decoded exact RGB subarray of a pinned source.
    Crop boundary guards remain unchanged. Only a proven translation is added.
    """
    protocol=read(context_protocol)
    entries=[case for case in protocol['cases']
             if Path(case['fresh_run_directory']).resolve()==Path(origin['directory'])]
    if len(entries)!=1:
        raise ValueError('context protocol must uniquely identify this run')
    entry=entries[0]
    binding_keys=['image_sha256','image_size','coordinate_frame']
    crop_binding={key:entry['crop_binding'][key] for key in binding_keys}
    if origin['image_binding']!=crop_binding:
        raise ValueError('context crop fingerprint/frame mismatch')
    source=Path(entry['source_binding']['image_path']).resolve()
    full_binding=image_binding(source)
    if full_binding!={key:entry['source_binding'][key] for key in binding_keys}:
        raise ValueError('context original fingerprint/frame mismatch')
    box=entry['crop_box_xyxy']
    width,height=full_binding['image_size']
    if (len(box)!=4 or any(type(v) is not int for v in box)
        or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)):
        raise ValueError('invalid context crop coordinates')
    with Image.open(source) as opened:
        crop=np.asarray(opened.convert('RGB').crop(box))
    with Image.open(origin['image_path']) as opened:
        actual=np.asarray(opened.convert('RGB'))
    if crop.shape!=actual.shape or not np.array_equal(crop,actual):
        raise ValueError('crop pixels differ from the declared original region')
    lifted=deepcopy(records)
    for record in lifted:
        for key in ['tips_xy','path_xy']:
            record['geometry'][key]=[[x+box[0],y+box[1]] for x,y in record['geometry'][key]]
        record['geometry_coordinate_frame']='verified_crop_pixels_translated_to_original_source'
        # Never clear boundary_truncated after embedding a crop in a larger image.
    meta=deepcopy(origin)
    meta.update(mask_source_image_path=origin['image_path'],image_path=str(source),
                frame_binding=full_binding,context_protocol_sha256=sha256(context_protocol),
                context_protocol_path=str(Path(context_protocol).resolve()),
                verified_crop_box_xyxy=box,crop_boundary_guards_preserved=True,
                coordinate_translation=[box[0],box[1]],no_crop_stitching=True)
    return meta,lifted


def frame_binding(origin):
    return origin.get('frame_binding',origin['image_binding'])


def load_run(directory):
    manifest = read(Path(directory)/'run_manifest.json')
    source = manifest['image_binding'].get('image_path') or read(Path(directory)/'sam/report.json')['input']
    origin = verified_run(directory, source)
    origin['image_path'] = str(Path(source).resolve())
    records = []
    for index, path in enumerate(origin['paths']):
        with Image.open(path) as opened:
            record = extract_mask(np.asarray(opened.convert('L')), origin['scores'][index], path.stem)
        record.update(source_mask_path=str(path), source_mask_sha256=sha256(path))
        records.append(record)
    origin['mask_files'] = [{'path':str(p), 'sha256':sha256(p)} for p in origin.pop('paths')]
    return origin, records


def read_bgr(path):
    with Image.open(path) as opened:
        return np.asarray(opened.convert('RGB'))[:,:,::-1].copy()


def render_pair(ref, ins, ref_records, ins_records, result, destination):
    with Image.open(ref['image_path']) as opened:
        left = opened.convert('RGB')
    with Image.open(ins['image_path']) as opened:
        right = opened.convert('RGB')
    ref_map = {r['record_id']:r for r in ref_records}
    ins_map = {r['record_id']:r for r in ins_records}
    for side, lookup, key in [(left,ref_map,'reference_record_id'), (right,ins_map,'inspection_record_id')]:
        draw = ImageDraw.Draw(side)
        for index, item in enumerate(result['proposals'],1):
            record = lookup[item[key]]
            draw.line([tuple(p) for p in record['geometry']['path_xy']], fill='#008c7f', width=2)
            for x,y in record['geometry']['tips_xy']:
                draw.ellipse((x-3,y-3,x+3,y+3), outline='#d67500', width=2)
            draw.text(tuple(record['geometry']['tips_xy'][0]),str(index),fill='#b30000')
    # Separate native frames; never draw registered coordinates onto the source.
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    canvas = Image.new('RGB',(left.width+right.width,max(left.height,right.height)+85),'white')
    canvas.paste(left,(0,85));canvas.paste(right,(left.width,85))
    draw=ImageDraw.Draw(canvas)
    draw.text((8,5),'可见线段关联候选；不是已确认接线',font=font,fill='#263342')
    draw.text((8,30),f"候选 {len(result['proposals'])}，电气连接仍未确认",font=font,fill='#263342')
    draw.text((8,57),'参考源像素',font=font,fill='#263342')
    draw.text((left.width+8,57),'待检源像素',font=font,fill='#263342')
    canvas.save(destination)


def execute(reference_run, inspection_run, output, context_protocol=None):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve existing outputs')
    ref, ref_records = load_run(reference_run)
    ins, ins_records = load_run(inspection_run)
    if context_protocol:
        ref,ref_records=lift_verified_context(ref,ref_records,context_protocol)
        ins,ins_records=lift_verified_context(ins,ins_records,context_protocol)
    files = [Path(__file__),Path(__file__).with_name('core.py'),
             Path(__file__).with_name('run_review.py'),Path(__file__).with_name('paired_evidence.py'),
             PROJECT/'prototype/assembly_auto_review_robust_v3.py',
             PROJECT/'prototype/assembly_auto_review_robust_v2.py',
             PROJECT/'prototype/assembly_auto_review_dino.py']
    pins={str(p):sha256(p) for p in files}
    output.mkdir(parents=True,exist_ok=False)
    save(output/'protocol.json', {'created_at':datetime.now(timezone.utc).isoformat(),
         'reference':ref,'inspection':ins,'source_pins':pins,
         'endpoint_radius_reference_diagonal_ratio':.005,'minimum_mask_score':.75,
         'registration_gates':'unchanged mainline automatic_homography',
         'fresh_SAM_inference':False,'no_port_confirmation':True,'no_GT_reads':True})
    save(output/'progress.json',{'status':'registration'})
    started=time.perf_counter()
    if frame_binding(ref)==frame_binding(ins):
        transform=np.eye(3)
        registration={'method':'exact_source_hash_and_frame_identity_control',
                      'alignment_quality':{'reliable':True}, 'self_control':True,
                      'source_to_reference_homography':transform.tolist()}
    else:
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(PROJECT/'prototype'))
        from assembly_auto_review_robust_v3 import automatic_homography
        _,registration=automatic_homography(read_bgr(ref['image_path']),
                                           read_bgr(ins['image_path']))
        transform=np.asarray(registration.get('source_to_reference_homography',np.eye(3)))
    result=propose_correspondences(ref_records,ins_records,transform,
                                  registration['alignment_quality'],frame_binding(ref)['image_size'])
    result.update(status='complete',seconds=time.perf_counter()-started,
                  registration=registration,reference=ref,inspection=ins,
                  reference_records=ref_records,inspection_records=ins_records,
                  fresh_SAM_inference=False,topology_decision='insufficient_evidence',
                  deployed=False,new_confirmed_connections=0)
    for directory,previous in [(reference_run,ref),(inspection_run,ins)]:
        current=verified_run(directory,previous.get('mask_source_image_path',previous['image_path']))
        if current['manifest_sha256']!=previous['manifest_sha256']:
            raise ValueError('SAM input drift during correspondence')
        if image_binding(previous['image_path'])!=frame_binding(previous):
            raise ValueError('full-frame context changed during correspondence')
        if context_protocol and sha256(context_protocol)!=previous['context_protocol_sha256']:
            raise ValueError('context protocol changed during correspondence')
    if any(sha256(p)!=digest for p,digest in pins.items()):
        raise ValueError('runner/mainline drift during correspondence')
    result['sources_and_code_unchanged']=True
    result['protocol_sha256']=sha256(output/'protocol.json')
    render_pair(ref,ins,ref_records,ins_records,result,output/'paired_paths.png')
    save(output/'report.json',result)
    save(output/'progress.json',{'status':'complete','proposals':len(result['proposals'])})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--inspection-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--context-protocol',type=Path)
    args=parser.parse_args()
    existed=args.output.exists()
    try:
        r=execute(args.reference_run,args.inspection_run,args.output,args.context_protocol)
    except Exception as error:
        if not existed and args.output.is_dir():
            save(args.output/'progress.json',{'status':'failed','type':type(error).__name__,'reason':str(error)})
        raise
    print(json.dumps({k:r[k] for k in ['decision','registration','new_confirmed_connections','sources_and_code_unchanged']},ensure_ascii=False))
    print('proposal_count='+str(len(r['proposals'])))


if __name__=='__main__':
    main()
