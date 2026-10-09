"""Pin fixed original cases and dependencies BEFORE fresh inference."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from run_audit import source_pins
from run_paired_evidence import load_run
from run_prompt_contrast import digest, save

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('E:/PythonProject10')


def main():
    import psutil
    available = psutil.virtual_memory().available
    if available < 6 * 2**30:
        raise RuntimeError('less than 6 GiB available before SAM build')
    out = ROOT / 'artifacts/route_prompt_contrast_20261005'
    if out.exists():
        raise FileExistsError('preserve existing experiment')
    cases = []
    for name, relative in [('full_cabinet_1','bound_sam_topology_20261001/cabinet_1_fresh'),
                           ('crop_cabinet_1','sam_crop_coverage_20261001/cabinet_1_crop_run'),
                           ('crop_cabinet_2','sam_crop_coverage_20261001/cabinet_2_crop_run')]:
        origin, _ = load_run(ROOT / 'artifacts' / relative)
        cases.append({'id':name,'historical_run':origin['directory'],
                      'historical_manifest_sha256':origin['manifest_sha256'],
                      'source':dict(origin['image_binding'],path=origin['image_path'])})
    checkpoint = PROJECT / 'models/sam3/sam3.pt'
    sam_source = PROJECT / 'runtime/sam3/source'
    files = [Path(__file__),Path(__file__).with_name('run_prompt_contrast.py'),
             Path(__file__).with_name('analyze_prompt_contrast.py'),
             *[Path(__file__).with_name(n) for n in ['core.py','run_review.py','run_paired_evidence.py','run_audit.py']],
             checkpoint, *sorted((sam_source / 'sam3').rglob('*.py'))]
    pins = {str(path.resolve()):digest(path) for path in files}
    out.mkdir(parents=True,exist_ok=False)
    save(out / 'protocol.json', {'created_at':datetime.now(timezone.utc).isoformat(),
         'cases':cases,'prompts':['cable','wire'],'retrieval_threshold':.5,
         'whole_mask_geometry_threshold':.75,'checkpoint':str(checkpoint),
         'sam_source':str(sam_source),'pins':pins,'mainline_pins':source_pins(),
         'preflight_available_GiB':available/2**30,'one_serial_model_process':True,
         'image_encoder_cache_reused':False,'model_observer_count':1,
         'no_GT_or_terminal_confirmation':True,'no_boundary_clear_or_fragment_split':True,
         'stop_if':'source/code drift, runtime failure or between-case time budget >30 minutes',
         'decision':'research only; never swap mainline based on counts or confidence',
         'evaluation':'all whole masks; exact baseline reproducibility; IoU diagnostic coverage of previous masks; visual review; no physical connectivity accuracy claim'})
    print(json.dumps({'protocol':str(out / 'protocol.json'),'cases':len(cases),
                      'available_GiB':round(available/2**30,2)},ensure_ascii=False))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
