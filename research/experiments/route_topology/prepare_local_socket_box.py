import json
from pathlib import Path
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from local_socket_box import positive_box

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/local_socket_box_source_20261008'


def main():
    prior=ROOT/'artifacts/local_socket_crop_source_20261008/protocol.json'
    p=json.loads(prior.read_text(encoding='utf-8'));verify(p['pins']);assert source_pins()==p['mainline_pins']
    scope=json.loads(Path(p['confirmed_scope_path']).read_text(encoding='utf-8'));anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    for case in p['cases']:
        pose=next(a for a in case['anchors'] if a['id']=='FAN_CPU')
        box,norm=positive_box(anchor['bbox_xyxy'],pose['inspection_to_reference_local'],case['crop_box_xyxy'])
        case['positive_box_source_xyxy']=box;case['positive_box_cxcywh_normalized']=norm
    files=[Path(__file__),Path(__file__).with_name('local_socket_box.py'),Path(__file__).with_name('run_local_socket_box.py'),
           Path(__file__).with_name('analyze_local_socket_box.py'),Path(__file__).with_name('audit_local_socket_crop.py'),
           prior,ROOT/'artifacts/LOCAL_SOCKET_BOX_PROTOCOL_20261008.md']
    p['pins'].update({str(f):digest(f) for f in files});p.update(positive_reference_padding=.5,recipes=['cable','cable_plus_local_socket_box'],no_geometry_prompt=False)
    OUT.mkdir(exist_ok=False);verify(p['pins']);save(OUT/'protocol.json',p)
    print('prepared five fixed crops/spatial boxes; no SAM yet')


if __name__=='__main__':main()
