"""Prepare an actual-photo reference review page; NEVER forge human confirmation."""
import base64
import json
from pathlib import Path

from core import image_binding, sha256
from run_review import save
from visible_lead_scope import validate_scope

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_visible_fan_scope_20261005'
    context=json.loads((evidence/'context_protocol.json').read_text(encoding='utf-8'))['cases'][0]
    source=Path(context['source_binding']['image_path'])
    binding=image_binding(source)
    if binding!={k:context['source_binding'][k] for k in binding}:
        raise ValueError('reference source drift')
    crop=evidence/'reference_scope.png'
    if image_binding(crop)!={k:context['crop_binding'][k] for k in binding}:
        raise ValueError('reference crop drift')
    scope={'schema_version':1,'kind':'reference_once_visible_lead_attachment',
        'reference_image_path':str(source),'reference_binding':binding,
        'scope_description':'CPU fan visible lead emergence to motherboard FAN_CPU socket; internal fan terminal hidden',
        # Draft anatomy annotations from actual raw reference, not GT or SAM tips.
        'anchors':[{'id':'FAN_LEAD','kind':'visible_lead_emergence','bbox_xyxy':[1480,1160,1535,1220]},
                   {'id':'FAN_CPU','kind':'wire_entry_socket','bbox_xyxy':[1600,1000,1700,1050]}],
        'expected_visible_attachment':['FAN_CPU','FAN_LEAD'],
        'reference_review':{'confirmed':False,'reviewer':None,'evidence_note':None},
        'annotation_origin':'assistant_reference_photo_draft_pending_real_human_review',
        'electrical_terminal_pair_established':False}
    validate_scope(scope,binding)
    payload={'scope':scope,'crop_box_xyxy':context['crop_box_xyxy'],
        'crop_data_url':'data:image/png;base64,'+base64.b64encode(crop.read_bytes()).decode()}
    template=Path(__file__).with_name('reference_lead_calibration.html')
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    output=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005'
    output.mkdir(parents=True,exist_ok=False)
    (output/'index.html').write_text(template.read_text(encoding='utf-8').replace('__REFERENCE_LEAD_PAYLOAD__',encoded),encoding='utf-8')
    save(output/'reference_scope_draft.json',scope)
    save(output/'report.json',{'status':'complete','reference_human_confirmed':False,
        'confirmed_real_connections':0,'fresh_SAM_inference_this_builder':False,
        'reference_binding':binding,'calibration_page_sha256':sha256(output/'index.html'),
        'draft_sha256':sha256(output/'reference_scope_draft.json'),
        'source_pins':{str(p):sha256(p) for p in [source,crop,template,Path(__file__)]}})
    print(output/'index.html')


if __name__=='__main__':main()
