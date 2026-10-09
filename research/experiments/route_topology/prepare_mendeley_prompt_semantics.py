"""Freeze wire+box and image-only box alternatives without tuning inspection ROIs."""
import json
from pathlib import Path

from PIL import Image

from core import image_binding
from run_audit import source_pins
from run_prompt_contrast import digest, save, verify

ROOT=Path(__file__).resolve().parents[2]


def main():
    previous=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005'
    old=json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((previous/'inference_report.json').read_text(encoding='utf-8'))
    if report['status']!='complete':raise ValueError('protect preceding inference')
    verify(old['pins'])
    if source_pins()!=old['mainline_pins']:raise ValueError('mainline drift')
    output=ROOT/'artifacts/mendeley_prompt_semantics_20261005'
    if output.exists():raise FileExistsError('new experiment only')
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for case in old['cases']:
        original=case['original_source'];source=Path(original['path'])
        if image_binding(source)!={k:original[k] for k in ['image_sha256','image_size','coordinate_frame']}:
            raise ValueError('original source drift')
        destination=output/(case['id']+'_original_crop.png')
        with Image.open(source) as opened:opened.convert('RGB').crop(case['crop_box_xyxy']).save(destination)
        cases.append({**case,'source':{'path':str(destination),**image_binding(destination)}})
    files=[Path(__file__),Path(__file__).with_name('run_mendeley_prompt_semantics.py'),
        previous/'protocol.json',previous/'inference_report.json',*[Path(c['source']['path']) for c in cases]]
    save(output/'protocol.json',{**old,'cases':cases,
        'pins':{**old['pins'],**{str(p.resolve()):digest(p) for p in files}},
        'recipes':['wire_plus_reference_anatomy_box','visual_reference_anatomy_box'],
        'minimum_score':.75,'candidate_only':True,
        'motivation':'cable+box found reference bundle but inspection mask disconnected; test two official prompt semantics',
        'stop_if':'source/code drift, runtime failure or30min between-case; no retries/threshold scans',
        'posthoc_development_experiment':True})
    print(output/'protocol.json')


if __name__=='__main__':main()
