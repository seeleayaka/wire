"""Fixed90CCW acquisition; baseline qualified poses/appearance explicitly shared."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256,image_binding
from run_prompt_contrast import verify,save
from run_audit import source_pins
from quarter_turn_view import forward,prompt

ROOT=Path(__file__).resolve().parents[2]


def main():
    src=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    ap=ROOT/'artifacts/mendeley_cable_socket_controls_audit_20261006/report.json'
    audit=json.loads(ap.read_text(encoding='utf-8'))
    assert audit['status']=='PASS' and audit['source_report_sha256']==sha256(src/'report.json')
    base=json.loads((src/'protocol.json').read_text(encoding='utf-8'));verify(base['pins'])
    before=source_pins();assert before==base['mainline_pins']
    out=ROOT/'artifacts/mendeley_quarter_turn_controls_20261007';out.mkdir(exist_ok=False)
    rows=json.loads((src/'preparation_report.json').read_text(encoding='utf-8'))['cases']
    assert [r['id'] for r in rows]==['reference','source_visible_01','source_visible_02','source_exposed_01','source_exposed_02']
    pins={**base['pins'],**{str(p):sha256(p) for p in [Path(__file__),ap,src/'protocol.json',src/'report.json',
        *[Path(__file__).with_name(n+'.py') for n in ['quarter_turn_view','run_quarter_turn_controls','audit_quarter_turn_case']] ]}}
    prepared=[]
    for row in rows:
        origin=row['original_source'];context=row['crop_context']
        assert sha256(origin['path'])==origin['image_sha256']
        native=np.asarray(Image.open(origin['path']).convert('RGB').crop(context['crop_box_xyxy']))
        assert np.array_equal(native,np.asarray(Image.open(context['source']['path']).convert('RGB')))
        original_path=out/(row['id']+'_original_crop.png'); Image.fromarray(native).save(original_path)
        rotated_path=out/(row['id']+'_rotated_crop.png'); Image.fromarray(forward(native)).save(rotated_path)
        pins[str(original_path)]=sha256(original_path);pins[str(rotated_path)]=sha256(rotated_path)
        transformed=dict(context,source=dict(path=str(rotated_path),**image_binding(rotated_path)),
            original_crop_path=str(original_path),original_crop_size=[native.shape[1],native.shape[0]],
            positive_box_cxcywh_normalized=prompt(context['positive_box_cxcywh_normalized']),
            exact_input_rotation='90CCW_pixel_permutation',coordinate_restore='mask90CW_pixel_permutation')
        prepared.append(dict(row,crop_context=transformed))
    save(out/'preparation_report.json',dict(status='complete',cases=prepared,
        original_decodes_fresh=True,poses_and_socket_appearance_explicitly_reused=True))
    pins[str(out/'preparation_report.json')]=sha256(out/'preparation_report.json')
    save(out/'protocol.json',dict(base,cases=[r['crop_context'] for r in prepared],pins=pins,
        baseline_report=str(src/'report.json'),pixel_view='90CCW_exact',
        reference_first_stop_if_not_unique_native_two_anchor=True,
        visible01_second_stop_if_no_new_native_gain=True,
        no_alternate_angle_or_prompt_search=True,all_five_planned_ids=[r['id'] for r in rows],
        no_interpolation_bridging_union_or_extra_observer=True,recipe_is_prompt_not_view=True,
        actual_original_RGB_review_required=True,no_automatic_demo_extension=True,deployed=False))
    print('prepared5 fixed controls; reference-first, exact native inverse, no extra model vote')


if __name__=='__main__':main()
