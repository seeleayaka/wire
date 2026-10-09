import json
from pathlib import Path
from identity_visual_review_contract import compare_review
from run_prompt_contrast import save,digest,verify
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/wire_identity_photo_review_20261008'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));verify(manifest['pins'])
    before=source_pins();assert before==manifest['mainline_pins']
    original=ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008/report.json'
    files=[OUT/'ai_visual_draft.json',OUT/'manifest.json',original,Path(__file__),Path(__file__).with_name('identity_visual_review_contract.py')]
    pins={str(p):digest(p) for p in files}
    draft=json.loads((OUT/'ai_visual_draft.json').read_text(encoding='utf-8'))
    endpoint=json.loads(original.read_text(encoding='utf-8'))
    report=compare_review(draft,manifest,endpoint)
    report.update(status='complete_AI_visual_comparison_not_physical_acceptance',pins=pins,
        fresh_SAM_calls=0,not_deployed=True,not_blind_evaluation=True,
        visible_route_but_old_native_masks_nearly_empty=['case_14','case_15'],
        same_color_two_endpoint_competition_example_count=0)
    for sample in manifest['samples']:assert digest(sample['source_path'])==sample['source_sha256']
    verify(pins);assert source_pins()==before
    save(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','pins']}))


if __name__=='__main__':main()
