"""Source-only, fixed color/edge agreement trial AFTER failed combined classifier.

Explicitly reuses independently reconstructed source descriptors and the source
calibration development set, not a new heldout accuracy measurement.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
from core import sha256
from run_audit import source_pins
from run_review import save
from socket_phenotype import fit,probability,predict,source_gate,POLICY as MODEL_POLICY
from socket_phenotype_agreement import feature_view,agreement,POLICY

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005'
    audit_path=ROOT/'artifacts/mendeley_socket_source_phenotype_audit_20261005/report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status']!='PASS':raise ValueError('source pixel/optimisation audit not passed')
    old_protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    old=json.loads((source/'report.json').read_text(encoding='utf-8'))
    if any(sha256(p)!=d for p,d in audit['pins'].items()) or source_pins()!=old_protocol['mainline_pins']:
        raise ValueError('audited source/E drift')
    files=[Path(__file__),Path(__file__).with_name('socket_phenotype.py'),Path(__file__).with_name('socket_phenotype_agreement.py'),source/'report.json',source/'protocol.json',audit_path]
    pins={str(p):sha256(p) for p in files};mainline=source_pins()
    output=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005'
    output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),'policy':POLICY,
        'model_policy':MODEL_POLICY,'pins':pins,'mainline_pins':mainline,'source_protocol':old_protocol,
        'original_images_decoded_fresh':False,'audited_source_features_reused':True,
        'inspection_inputs_or_labels_read':False,'no_label_or_GT_change':True,
        'stop_if':'source gate fails; no inspection or further same-calibration feature search'})
    begun=time.perf_counter();inventory={r['id']:r for r in old['prepared']}
    fitting=[inventory[k] for k in old_protocol['fit_ids']];calib=[inventory[k] for k in old_protocol['calibration_ids']]
    models={};individual={}
    for view in POLICY['views']:
        model=fit([feature_view(r['descriptor'],view) for r in fitting],[r['visual_label'] for r in fitting])
        scores={k:[probability(model,feature_view(r['descriptor'],view)) if k==0 else 1-probability(model,feature_view(r['descriptor'],view))
            for r in calib if r['visual_label']==k] for k in [0,1]}
        individual[view]={r['id']:predict(model,scores,feature_view(r['descriptor'],view)) for r in calib}
        models[view]={**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()},
            'calibration':{str(k):v for k,v in scores.items()}}
    rows=[{'id':r['id'],'visual_label':r['visual_label'],
        'prediction':agreement({view:individual[view][r['id']] for view in POLICY['views']})} for r in calib]
    gate=source_gate(rows)
    if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=mainline:raise ValueError('source/code/E drift')
    save(output/'model.json',models)
    result={'status':'complete','source_gate':gate,'calibration_results':rows,
        'individual_view_gates':{view:source_gate([{'visual_label':r['visual_label'],'prediction':individual[view][r['id']]} for r in calib]) for view in POLICY['views']},
        'seconds':time.perf_counter()-begun,'inspection_inputs_or_labels_read':False,
        'field_accuracy_not_measured':True,'reused_development_calibration':True,
        'new_confirmed_connections':0,'reference_review_confirmed':False,'deployed':False,
        'protocol_sha256':sha256(output/'protocol.json')}
    save(output/'report.json',result);print(json.dumps({k:result[k] for k in ['status','source_gate','individual_view_gates','seconds']}))


if __name__=='__main__':main()
