"""Offline original-mask/path review -> unconfirmed paired-map draft.

Embedding pixels is local only. The page never invents port rectangles,
expected edges, human approvals or physical connections from mask endpoints.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from core import sha256, image_binding
from run_paired_evidence import load_run
from run_review import verified_run
from run_review import save


def execute(reference_run,inspection_run,output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve existing workbenches')
    payload={}
    origins={}
    pins={str(Path(__file__).resolve()):sha256(__file__),
          str(Path(__file__).with_name('mask_workbench.html')):sha256(Path(__file__).with_name('mask_workbench.html'))}
    for key,run in [('reference',reference_run),('inspection',inspection_run)]:
        origin,records=load_run(run)
        origins[key]=origin
        path=Path(origin['image_path'])
        mime='image/png' if path.suffix.lower()=='.png' else 'image/jpeg'
        source_bytes=path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest()!=origin['image_binding']['image_sha256']:
            raise ValueError('embedded source bytes changed after verification')
        masks=[]
        for index,item in enumerate(origin['mask_files']):
            pixels=Path(item['path']).read_bytes()
            if hashlib.sha256(pixels).hexdigest()!=item['sha256']:
                raise ValueError('embedded mask bytes changed after verification')
            masks.append({'id':index+1,'sha256':item['sha256'],
                          'data_url':'data:image/png;base64,'+base64.b64encode(pixels).decode()})
        payload[key]={'image_path':str(path),'image_binding':origin['image_binding'],
                      'origin_manifest_sha256':origin['manifest_sha256'],
                      'origin_evidence_kind':origin['origin_evidence_kind'],'records':records,
                      'data_url':f'data:{mime};base64,'+base64.b64encode(source_bytes).decode(),
                      'masks':masks}
    template=Path(__file__).with_name('mask_workbench.html').read_text(encoding='utf-8')
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    for origin in origins.values():
        checked=verified_run(origin['directory'],origin['image_path'])
        if checked['manifest_sha256']!=origin['manifest_sha256']:
            raise ValueError('origin manifest changed during workbench build')
    if any(sha256(p)!=digest for p,digest in pins.items()):
        raise ValueError('workbench runner/template changed during build')
    output.mkdir(parents=True,exist_ok=False)
    (output/'index.html').write_text(template.replace('__EMBEDDED_EVIDENCE__',encoded),encoding='utf-8')
    # Make this tool's reuse and evidence type explicit, not "fresh inference".
    save(output/'report.json',{'status':'complete','fresh_SAM_inference':False,
         'new_confirmed_connections':0,'confirmed_ports':0,'source_pins':pins,
         'origins':{k:{field:v[field] for field in ['image_path','image_binding','origin_manifest_sha256','origin_evidence_kind']} for k,v in payload.items()},
         'workbench_sha256':sha256(output/'index.html'),'electrical_continuity':'not_assessed'})
    return output/'index.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--inspection-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(execute(args.reference_run,args.inspection_run,args.output))
