"""One new exposed-socket negative control, no further prompting/threshold sweep."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256,image_binding
from prepare_mendeley_scope import inspection_scope
from run_audit import source_pins
from run_prompt_contrast import verify,save

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_phenotype_fresh30_20261005'
    previous=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005'
    old=json.loads((previous/'protocol.json').read_text(encoding='utf-8'));verify(old['pins'])
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    audit_path=ROOT/'artifacts/mendeley_socket_phenotype_fresh30_audit_20261005/report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status']!='PASS' or any(sha256(p)!=d for p,d in audit['pins'].items()):raise ValueError('fresh socket evidence audit required')
    row=next(r for r in report['cases'] if r['id']=='case_06')
    if row['phenotype']!='socket_contacts_exposed':raise ValueError('negative evidence no longer matches frozen sample selection')
    path=Path(row['input_path']);binding=image_binding(path);matrix=np.array(row['registration']['source_to_reference_homography'])
    crop_box=inspection_scope([1380,870,1780,1270],matrix,binding['image_size'])
    anatomy=inspection_scope([1420,900,1730,1230],matrix,binding['image_size'])
    output=ROOT/'artifacts/mendeley_exposed_socket_sam_20261005';output.mkdir(exist_ok=False)
    with Image.open(path) as im:crop=im.convert('RGB').crop(crop_box)
    crop_path=output/'case_06_original_crop.png';crop.save(crop_path)
    local=np.array(anatomy)-np.array(crop_box[:2]*2)
    if not (0<=local[0]<local[2]<=crop.width and 0<=local[1]<local[3]<=crop.height):raise ValueError('anatomy outside crop, never clip')
    box=[float((local[0]+local[2])/(2*crop.width)),float((local[1]+local[3])/(2*crop.height)),float((local[2]-local[0])/crop.width),float((local[3]-local[1])/crop.height)]
    case={'id':'case_06','source':{'path':str(crop_path),**image_binding(crop_path)},
        'original_source':{'path':str(path),**binding},'crop_box_xyxy':crop_box,
        'positive_box_cxcywh_normalized':box,'positive_box_source_xyxy':anatomy,
        'socket_phenotype':'socket_contacts_exposed','FAN_LEAD_local_spatial_gate_passed':False}
    files=[Path(__file__),Path(__file__).with_name('run_mendeley_geometry.py'),Path(__file__).with_name('run_prompt_contrast.py'),Path(__file__).with_name('prepare_mendeley_scope.py'),source/'report.json',audit_path,path,crop_path]
    pins={**old['pins'],**{str(p):sha256(p) for p in files}}
    save(output/'protocol.json',{'cases':[case],'pins':pins,'mainline_pins':source_pins(),
        'sam_source':old['sam_source'],'checkpoint':old['checkpoint'],
        'recipes':['cable','cable_plus_reference_anatomy_box'],'retrieval_threshold':.5,'whole_mask_geometry_threshold':.75,
        'reference_anatomy_xyxy':[1420,900,1730,1230],'fresh_image_encoder_required':True,
        'original_RGB_crop_no_pixel_warp':True,'previous_fresh_global_registration_explicitly_reused':True,
        'prior_positive_reference_and_reroute_runs_reused_only_for_comparison':True,
        'model_observer_count':1,'GT_read':False,'reference_review_confirmed':False,
        'not_independent_accuracy_test':True,'sample_selected_after_candidate_preview':True,
        'stop_if':'source/code drift or runtime failure; no changed crop/threshold/prompt sweep'})
    print(output/'protocol.json')


if __name__=='__main__':main()
