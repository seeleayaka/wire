"""Independent support/threshold/decision replay from saved source pixels."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    src=ROOT/'artifacts/mendeley_positive_contact_source_20261006'
    protocol=json.loads((src/'protocol.json').read_text(encoding='utf-8'))
    assert source_pins()==protocol['mainline_pins']
    assert all(sha256(p)==d for p,d in protocol['pins'].items())
    split=json.loads((ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json').read_text(encoding='utf-8'))
    labels=json.loads((ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json').read_text(encoding='utf-8'))
    labels={r['id']:r['visual_label'] for r in labels['rows']}
    masks={}
    for i in split['fit_ids']+split['calibration_ids']:
        p=ROOT/'artifacts/mendeley_component_color_source_20261006'/(i+'_fresh_socket.png')
        hsv=cv2.cvtColor(np.asarray(Image.open(p).convert('RGB')),cv2.COLOR_RGB2HSV)
        masks[i]=(hsv[:,:,1]<80)&(hsv[:,:,2]>=160)
    p0=np.mean([masks[i] for i in split['fit_ids'] if labels[i]==0],axis=0)
    p1=np.mean([masks[i] for i in split['fit_ids'] if labels[i]==1],axis=0)
    support=(p0>=.6)&(p0-p1>=.4)
    assert np.array_equal(support,np.asarray(Image.open(src/'fit_contact_support.png'))>0)
    scores={i:float(m[support].mean()) for i,m in masks.items()}
    threshold=max(np.quantile([scores[i] for i in split['fit_ids'] if labels[i]==1],.95),np.quantile([scores[i] for i in split['fit_ids'] if labels[i]==0],.05))
    baseline=json.loads((ROOT/'artifacts/mendeley_component_color_source_20261006/report.json').read_text(encoding='utf-8'))['baseline_LOO_results']
    rescue={r['id']:r for r in json.loads((ROOT/'artifacts/mendeley_component_rescue_source_20261006/report.json').read_text(encoding='utf-8'))['cases']}
    actual=json.loads((src/'report.json').read_text(encoding='utf-8'))
    results={r['id']:r for r in actual['cases']}
    assert threshold==actual['fit_threshold'] and int(support.sum())==actual['support_pixels']
    accepted=[];gains=[];losses=[]
    for r in baseline:
        i=r['id'];old=r['prediction']['visual_label_candidate']
        proposed=rescue[i]['prediction']['visual_label_candidate']
        expected=old if old is not None else (0 if proposed==0 and scores[i]>threshold and support.sum()>=6 else None)
        assert expected==results[i]['prediction']['visual_label_candidate']
        assert scores[i]==results[i]['positive_contact_score']
        if expected is not None:accepted.append((i,expected,labels[i]))
        if old is None and expected==labels[i]:gains.append(i)
        if old==labels[i] and expected!=old:losses.append(i)
    wrong=sum(a!=b for _,a,b in accepted)
    assert len(accepted)==74 and wrong==0 and gains==actual['source_gains'] and losses==[]
    out=ROOT/'artifacts/mendeley_positive_contact_audit_20261006';out.mkdir(exist_ok=False)
    report=dict(status='PASS',singleton_count=len(accepted),wrong=wrong,gains=gains,losses=losses,
        support_and_threshold_independently_recomputed=True,cached_source_pixels_explicit=True,
        classifier_scores_from_prior_reports_not_independently_refitted=True,
        source_gate_only_not_topology_acceptance=True,mainline_unchanged=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
