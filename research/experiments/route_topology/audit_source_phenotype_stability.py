"""Fixed FIT-source-only perturbation diagnostics, not a new accuracy test.

One-pixel registration offsets use real original pixels. Brightness copies are
explicit synthetic probes. No inspection data, fitting or threshold adjustment.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_review import save
from socket_appearance import descriptor
from socket_phenotype import predict
from socket_phenotype_agreement import feature_view,agreement

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    model_path=source/'model.json';raw=json.loads(model_path.read_text(encoding='utf-8'))
    labels_path=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    labels=json.loads(labels_path.read_text(encoding='utf-8'));inventory={r['id']:r for r in labels['rows']}
    ids=protocol['source_protocol']['fit_ids']
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    pins={str(p):sha256(p) for p in [Path(__file__),model_path,labels_path,source/'protocol.json',
        Path(__file__).with_name('socket_appearance.py'),Path(__file__).with_name('socket_phenotype.py'),
        Path(__file__).with_name('socket_phenotype_agreement.py')]}
    output=ROOT/'artifacts/mendeley_socket_source_stability_20261005';output.mkdir(exist_ok=False)
    probes={'original':(0,0,1.),'left1':(-1,0,1.),'right1':(1,0,1.),'up1':(0,-1,1.),'down1':(0,1,1.),
        'brightness90':(0,0,.9),'brightness110':(0,0,1.1)}
    save(output/'protocol.json',{'pins':pins,'source_fit_ids':ids,'probes':probes,
        'source_fit_only':True,'not_new_heldout_accuracy':True,'brightness_probes_synthetic':True,
        'classification_thresholds_unchanged':True,'no_model_fit_or_inspection_input':True,
        'stop_if':'source/input/E drift; never fix per-image results from this diagnostic'})
    models={view:({k:np.asarray(m[k]) if k in ['center','scale','weights'] else m[k] for k in ['center','scale','weights','bias']},
        {int(k):v for k,v in m['calibration'].items()}) for view,m in raw.items()}
    rows=[]
    for identity in ids:
        r=inventory[identity]
        if sha256(r['source_path'])!=r['source_sha256']:raise ValueError('source drift')
        with Image.open(r['source_path']) as im:rgb=np.asarray(im.convert('RGB'))
        h=np.array(r['pose']['inspection_to_reference_local']);results={}
        for name,(dx,dy,gain) in probes.items():
            transform=np.array([[1,0,-1600+dx],[0,1,-1000+dy],[0,0,1]])@h
            valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),transform,(100,50),flags=cv2.INTER_NEAREST)
            if not valid.all():raise ValueError('probe outside original pixels')
            patch=cv2.warpPerspective(rgb,transform,(100,50),flags=cv2.INTER_LINEAR)
            if gain!=1:patch=np.clip(patch.astype(float)*gain,0,255).astype(np.uint8)
            f=descriptor(patch);result=agreement({v:predict(m,c,feature_view(f,v)) for v,(m,c) in models.items()})
            results[name]=result['visual_label_candidate']
        emitted=[v for v in results.values() if v is not None]
        rows.append({'id':identity,'source_sha256':r['source_sha256'],'visual_source_label':r['visual_label'],
            'probe_predictions':results,'conflicting_singletons_across_probes':len(set(emitted))>1,
            'any_wrong_singleton_vs_qualitative_source_label':any(v!=r['visual_label'] for v in emitted),
            'any_abstention':any(v is None for v in results.values())})
    if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=protocol['mainline_pins']:raise ValueError('code/input/E drift')
    report={'status':'complete','source_fit_images':117,'source_probe_predictions':819,
        'photos_with_conflicting_probe_singletons':sum(r['conflicting_singletons_across_probes'] for r in rows),
        'photos_with_any_wrong_probe_singleton':sum(r['any_wrong_singleton_vs_qualitative_source_label'] for r in rows),
        'photos_with_any_abstention':sum(r['any_abstention'] for r in rows),'rows':rows,
        'new_confirmed_connections':0,'inspection_inputs_read':False,'not_field_or_new_heldout_accuracy':True,
        'source_annotation_not_human_GT':True,'model_or_threshold_changed':False}
    save(output/'report.json',report);print(json.dumps({k:report[k] for k in ['status','source_fit_images',
        'photos_with_conflicting_probe_singletons','photos_with_any_wrong_probe_singleton','photos_with_any_abstention']}))


if __name__=='__main__':main()
