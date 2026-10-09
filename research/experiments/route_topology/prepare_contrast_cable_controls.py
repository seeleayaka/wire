"""Change only segmentation photometry on all five fixed source controls."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256,image_binding
from run_prompt_contrast import verify,save
from run_audit import source_pins
from local_contrast_view import contrast_view,POLICY

ROOT=Path(__file__).resolve().parents[2]

def main():
    src=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    audit_path=ROOT/'artifacts/mendeley_cable_socket_controls_audit_20261006/report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'));assert audit['status']=='PASS'
    assert audit['source_report_sha256']==sha256(src/'report.json')
    base=json.loads((src/'protocol.json').read_text(encoding='utf-8'));verify(base['pins'])
    before=source_pins();assert before==base['mainline_pins']
    prepared=json.loads((src/'preparation_report.json').read_text(encoding='utf-8'))['cases']
    assert len(prepared)==5
    out=ROOT/'artifacts/mendeley_contrast_cable_controls_20261007';out.mkdir(exist_ok=False)
    pins={**base['pins'],**{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('local_contrast_view.py'),
        Path(__file__).with_name('run_contrast_cable_controls.py'),audit_path,src/'report.json',src/'protocol.json']}}
    rows=[];cases=[]
    for row in prepared:
        origin=row['original_source'];context=row['crop_context'];assert sha256(origin['path'])==origin['image_sha256']
        native=np.asarray(Image.open(origin['path']).convert('RGB').crop(context['crop_box_xyxy']))
        assert np.array_equal(native,np.asarray(Image.open(context['source']['path']).convert('RGB')))
        raw_path=out/(row['id']+'_original_crop.png');Image.fromarray(native).save(raw_path)
        view_path=out/(row['id']+'_contrast_crop.png');Image.fromarray(contrast_view(native)).save(view_path)
        for p in [raw_path,view_path]:pins[str(p)]=sha256(p)
        new_context=dict(context,source=dict(path=str(view_path),**image_binding(view_path)),
            original_crop_path=str(raw_path),photometric_view=POLICY,native_pixel_coordinates_preserved=True)
        rows.append(dict(row,crop_context=new_context,poses_and_socket_appearance_from_verified_baseline=True))
        cases.append(new_context)
    save(out/'preparation_report.json',dict(status='complete',cases=rows,
        only_originals_decoded_fresh=True,poses_and_DINO_explicitly_reused_for_one_variable_comparison=True))
    pins[str(out/'preparation_report.json')]=sha256(out/'preparation_report.json')
    save(out/'experiment_contract.json',dict(photometry=POLICY,baseline_report=str(src/'report.json'),baseline_audit=str(audit_path),
        all5_source_controls=True,no_demo_images=True,no_parameter_sweep=True,no_new_model_vote=True,
        preserve_old_supported_visible_cases=True,strict_new_visible_gain_required=True,
        exposed_high_socket_overlap_must_remain_zero=True,decision_thresholds_unchanged=True,
        all_predicted_mask_pixels_and_fragments_retained=True,no_bridge_or_pruning=True,
        source_gate_only_not_physical_continuity=True,no_automatic_demo_extension=True,deployed=False))
    pins[str(out/'experiment_contract.json')]=sha256(out/'experiment_contract.json')
    verify(pins);assert source_pins()==before
    save(out/'protocol.json',dict(base,cases=cases,pins=pins,photometry=POLICY,
        no_prior_SAM_reused=True,original_RGB_retained_for_actual_visual_review=True,deployed=False))
    print('prepared all five fixed source controls; no prior SAM reused')

if __name__=='__main__':main()
