"""Fresh original registration -> exact unwarped scope crops -> pinned SAM recipe.

Scope selected on reference device anatomy only, not labels. Inspection crop is
derived from the same reference scope via inverse homography. No GT reads.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

from core import image_binding
from prepare_reference_once import execute
from run_prompt_contrast import digest, save
from run_audit import source_pins

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('E:/PythonProject10')
DATA = PROJECT / 'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
OUT = ROOT / 'artifacts/mendeley_visible_fan_scope_20261005'


def inspection_scope(reference_box, inspection_to_reference, size):
    l, t, r, b = reference_box
    q = np.array([[l,t,1],[r,t,1],[r,b,1],[l,b,1]],float) @ np.linalg.inv(inspection_to_reference).T
    if np.any(np.abs(q[:,2])<1e-9) or not (np.all(q[:,2]>0) or np.all(q[:,2]<0)):
        raise ValueError('scope crosses projective horizon')
    xy = q[:,:2]/q[:,2:]
    box = [math.floor(xy[:,0].min()),math.floor(xy[:,1].min()),
           math.ceil(xy[:,0].max()),math.ceil(xy[:,1].max())]
    if not (0<=box[0]<box[2]<=size[0] and 0<=box[1]<box[3]<=size[1]):
        raise ValueError('scope outside inspection, no clipping')
    return box


def main():
    if OUT.exists():
        raise FileExistsError('new experiment only')
    import psutil
    if psutil.virtual_memory().available < 6*2**30:
        raise RuntimeError('less than 6GiB available before SAM')
    sources = {'reference': DATA/'train01/normal_073.JPG',
               'inspection': DATA/'test01/misrouted_001.JPG'}
    # Anatomy-defined development scope for visible CPU fan lead and FAN_CPU
    # socket, NOT a defect target box. Fan-side emergence may not expose its
    # actual electrical terminal: keep this explicit through every later stage.
    reference_box = [1380, 870, 1780, 1270]
    original_bindings = {k:image_binding(p) for k,p in sources.items()}
    registration = execute(sources['reference'],sources['inspection'],OUT/'registration')
    if not registration['registration']['alignment_quality']['reliable']:
        raise ValueError('scope requires unchanged production registration gates')
    matrix = np.array(registration['registration']['source_to_reference_homography'])
    boxes = {'reference':reference_box, 'inspection':inspection_scope(reference_box,matrix,
                                                    original_bindings['inspection']['image_size'])}
    cases, context = [], []
    for key,path in sources.items():
        with Image.open(path) as opened:
            crop = opened.convert('RGB').crop(boxes[key])
        destination = OUT/(key+'_scope.png')
        crop.save(destination)
        binding = image_binding(destination)
        run_dir = OUT/key/'cable'
        cases.append({'id':key, 'source':{'path':str(destination.resolve()),**binding}})
        context.append({'fresh_run_directory':str(run_dir.resolve()),
            'source_binding':{'image_path':str(path.resolve()),**original_bindings[key]},
            'crop_binding':{'image_path':str(destination.resolve()),**binding},
            'crop_box_xyxy':boxes[key]})
    if any(image_binding(sources[k]) != original_bindings[k] for k in sources):
        raise ValueError('original input drift')
    checkpoint = PROJECT/'models/sam3/sam3.pt'
    sam_source = PROJECT/'runtime/sam3/source'
    files = [Path(__file__),Path(__file__).with_name('run_prompt_contrast.py'),
        Path(__file__).with_name('prepare_reference_once.py'),Path(__file__).with_name('reference_once.py'),
        Path(__file__).with_name('core.py'),checkpoint,
        *sorted((sam_source/'sam3').rglob('*.py')), *sources.values(),
        *[Path(c['source']['path']) for c in cases]]
    pins = {str(path.resolve()):digest(path) for path in files}
    save(OUT/'context_protocol.json',{'cases':context,
        'reference_scope_definition':'visible CPU fan lead emergence and motherboard FAN_CPU socket',
        'inspection_scope_method':'inverse unchanged production homography, enclosing integer rectangle',
        'no_warped_SAM_input':True,'no_image_specific_inspection_annotation':True,
        'actual_fan_internal_terminal_visible':False})
    save(OUT/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'cases':cases,'prompts':['cable','wire'],'retrieval_threshold':.5,
        'checkpoint':str(checkpoint),'sam_source':str(sam_source),'pins':pins,
        'mainline_pins':source_pins(),'model_observer_count':1,
        'image_encoder_cache_reused':False,'whole_mask_geometry_threshold':.75,
        'original_sources':{k:{'path':str(p),**original_bindings[k]} for k,p in sources.items()},
        'reference_scope_xyxy':reference_box,'context_protocol_sha256':digest(OUT/'context_protocol.json'),
        'GT_read':False,'reference_calibration_confirmed':False,
        'no_topology_or_electrical_continuity_claim':True,
        'evaluation':'all whole masks, conservative geometry, boundary guards unchanged; visible lead evidence only',
        'stop_if':'input/code drift or runtime failure; no threshold or GT-guided crop scan'})
    print(json.dumps({'output':str(OUT),'cases':2,'prompts':2,'boxes':boxes},ensure_ascii=False))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
