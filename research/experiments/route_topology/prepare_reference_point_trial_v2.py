"""Fresh protocol for isolated CPU interactivity; preserve failed v1 exactly."""
from pathlib import Path
from PIL import Image
from run_review import read, save
from run_prompt_contrast import verify, digest
from bundle_runtime_pins import source_pins

ROOT = Path(__file__).resolve().parents[2]


def main():
    old = ROOT / 'artifacts/mendeley_reference_point_trial_20261006'
    p = read(old / 'protocol.json')
    verify(p['pins'])
    if source_pins() != p['mainline_pins']:
        raise ValueError('mainline drift')
    dependencies = ROOT / 'artifacts/interactive_cpu_dependencies_20261006'
    if not list(dependencies.glob('pycocotools/*.pyd')):
        raise ValueError('official binary dependency absent')
    output = ROOT / 'artifacts/mendeley_reference_point_trial_v2_20261006'
    output.mkdir(exist_ok=False)
    with Image.open(p['original_source']['path']) as image:
        image.convert('RGB').crop(p['crop_box_xyxy']).save(output / 'fresh_original_crop.png')
    p['source']['path'] = str(output / 'fresh_original_crop.png')
    p['source']['sha256'] = digest(output / 'fresh_original_crop.png')
    p['workspace_dependencies'] = str(dependencies)
    p['previous_failed_trial'] = str(old)
    p['cpu_edt_backend'] = 'opencv_precise_cpu_explicit_research_adapter'
    p['postprocess_max_hole_area'] = 0
    p['postprocess_max_sprinkle_area'] = 0
    files = [Path(__file__), Path(__file__).with_name('interactive_cpu_edt.py'),
             Path(__file__).with_name('run_reference_point_trial_v2.py'),
             Path(__file__).with_name('audit_reference_point_trial_v2.py'),
             output / 'fresh_original_crop.png', old / 'protocol.json']
    files += [f for f in dependencies.rglob('*') if f.is_file() and f.suffix != '.pyc']
    p['pins'].update({str(f): digest(f) for f in files})
    save(output / 'protocol.json', p)
    print('V2_PREPARED_FRESH_ORIGINAL_CROP_NO_MODEL_OR_EMBEDDING_REUSE')


if __name__ == '__main__':
    main()
