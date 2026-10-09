"""Audit original crop bytes, row bindings and provisional-only counts."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/wire_identity_photo_review_20261008'


def main():
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    manifest=read(OUT/'manifest.json');report=read(OUT/'report.json');draft=read(OUT/'ai_visual_draft.json')
    verify(manifest['pins']);verify(report['pins']);assert source_pins()==manifest['mainline_pins']
    old=read(ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008/report.json')
    old_by_id={r['id']:r for r in old['cases']};observations={r['review_id']:r for r in draft['observations']}
    assert len(observations)==len(manifest['samples'])==len(report['cases'])==30
    differences=[]
    for sample in manifest['samples']:
        assert digest(sample['source_path'])==sample['source_sha256']
        expected=np.asarray(Image.open(sample['source_path']).convert('RGB').crop(sample['crop_box_xyxy']))
        actual=np.asarray(Image.open(OUT/sample['detail_image']).convert('RGB'))
        assert np.array_equal(expected,actual)
        assert sample['source_sha256']==old_by_id[sample['case_id']]['source_binding']['image_sha256']
        observation=observations[sample['review_id']]
        if observation['socket']=='contacts_exposed_apparent' and old_by_id[sample['case_id']]['decision']!='reference_socket_exposure_observed':
            differences.append(sample['case_id'])
    assert Counter(r['socket'] for r in draft['observations'])==report['socket_observation_counts']
    assert differences==report['provisional_socket_difference_ids']==['case_07','case_08','case_09']
    assert report['accuracy_denominator']==report['physical_identity_GT_count']==report['human_confirmed_count']==0
    assert report['accuracy_rate'] is None and draft['human_confirmed'] is False
    result=dict(status='PASS',original_RGB_crops_byte_verified=30,source_SHA_bindings=30,
        original_case_decisions_unchanged=True,AI_description_is_not_independent_GT=True,
        provisional_difference_case_ids=differences,accepted_human_GT=0,
        report_sha256=digest(OUT/'report.json'),draft_sha256=digest(OUT/'ai_visual_draft.json'),
        no_second_visual_annotator=True,no_accuracy_or_calibration_claim=True)
    save(OUT/'audit_report.json',result);print(json.dumps(result))


if __name__=='__main__':main()
