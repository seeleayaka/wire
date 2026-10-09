"""Fresh raw-image registration and reference calibration draft; not a topology pass."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from core import image_binding, sha256
from reference_once import draft_template, map_reference_ports


def execute(reference, inspection, output, template_path=None):
    reference, inspection, output = map(Path, (reference, inspection, output))
    if output.exists():
        raise FileExistsError('preserve existing results')
    ref_binding, ins_binding = image_binding(reference), image_binding(inspection)
    template = (json.loads(Path(template_path).read_text(encoding='utf-8')) if template_path
                else draft_template(reference.resolve(), ref_binding))
    sources = [reference, inspection, Path(__file__), Path(__file__).with_name('reference_once.py'),
               Path(__file__).with_name('core.py')]
    project = Path('E:/PythonProject10')
    sources.extend(project / 'prototype' / name for name in (
        'assembly_auto_review_robust_v3.py', 'assembly_auto_review_robust_v2.py',
        'assembly_auto_review_dino.py'))
    if template_path:
        sources.append(Path(template_path))
    pins = {str(p.resolve()): sha256(p) for p in sources}
    started = time.perf_counter()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(project / 'prototype'))
    from assembly_auto_review_robust_v3 import automatic_homography
    with Image.open(reference) as opened:
        ref = np.asarray(opened.convert('RGB'))[:, :, ::-1].copy()
    with Image.open(inspection) as opened:
        ins = np.asarray(opened.convert('RGB'))[:, :, ::-1].copy()
    # Function expects reference first and returns INSPECTION -> REFERENCE.
    _, registration = automatic_homography(ref, ins)
    result = map_reference_ports(template, ref_binding, ins_binding,
        registration.get('source_to_reference_homography'), registration['alignment_quality'])
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError('image, template or code changed during preparation')
    output.mkdir(parents=True, exist_ok=False)
    result.update(status='complete', created_at=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter() - started, registration=registration,
        reference_binding=ref_binding, inspection_binding=ins_binding, source_pins=pins,
        inputs_read_from_original_images=True, previous_image_results_reused=False,
        fresh_SAM_inference=False, GT_read=False, deployed=False,
        calibration_stage_only=True)
    for name, value in [('report.json', result), ('reference_template.json', template)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reference', 'inspection', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--template', type=Path)
    args = parser.parse_args()
    result = execute(args.reference, args.inspection, args.output, args.template)
    print(json.dumps({key: result[key] for key in ['status', 'reason', 'seconds', 'registration']}, ensure_ascii=False))
