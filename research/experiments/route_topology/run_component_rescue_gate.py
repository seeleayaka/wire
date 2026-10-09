"""New frozen conservative compound source replay; no unseen accuracy claim."""
import json
from pathlib import Path
from core import sha256
from component_cues import gate
from component_rescue import rescue
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_component_rescue_source_20261006';out.mkdir(exist_ok=False)
    color_path=ROOT/'artifacts/mendeley_component_color_source_20261006/report.json'
    sem_path=ROOT/'artifacts/mendeley_component_semantic_source_v2_20261006/report.json'
    before=source_pins()
    pins={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('component_rescue.py'),color_path,sem_path]}
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,mainline_pins=before,
        rule='preserve every old supported label; only old unknown + colour singleton agrees semantic singleton can propose rescue',
        acceptance='zero wrong source singleton, zero old supported losses, strictly positive source gain, coverage>=.75',
        rule_designed_after_source_failure_analysis=True,source_calibration_reused_development=True,
        no_independent_field_test=True,one_image_source_not_two_independent_votes=True),indent=2),encoding='utf-8')
    color=json.loads(color_path.read_text(encoding='utf-8'));semantic=json.loads(sem_path.read_text(encoding='utf-8'))
    if color['baseline_LOO_results']!=semantic['baseline_LOO_results']:raise ValueError('baseline source replay differs')
    old=color['baseline_LOO_results'];c=color['calibration_results'];s=semantic['calibration_results']
    if [r['id'] for r in old]!=[r['id'] for r in c] or [r['id'] for r in c]!=[r['id'] for r in s]:raise ValueError('source inventory differs')
    rows=[];gains=[];losses=[]
    for baseline,cr,sr in zip(old,c,s):
        label=baseline['visual_label']
        if label!=cr['visual_label'] or label!=sr['visual_label']:raise ValueError('source labels drift')
        proposal=rescue(baseline['prediction'],cr['prediction'],sr['prediction'],True,True)
        rows.append(dict(id=baseline['id'],visual_label=label,prediction=proposal))
        if baseline['prediction']['visual_label_candidate']==label and proposal['visual_label_candidate']!=label:losses.append(baseline['id'])
        if baseline['prediction']['visual_label_candidate'] is None and proposal['visual_label_candidate']==label:gains.append(baseline['id'])
    result=gate(rows);passed=result['passed'] and not losses and len(gains)>0
    if before!=source_pins() or any(sha256(p)!=d for p,d in pins.items()):raise ValueError('E/code/input drift')
    report=dict(status='complete',source_gate=result,strict_net_source_gain=passed,
        baseline_gate=gate(old),source_gains=gains,source_losses=losses,cases=rows,
        reference_controls_required_before_any_inspection=True,source_annotation_not_human_GT=True,
        actual_gate_is_development_replay_not_independent_validation=True,
        mainline_unchanged=True,deployed=False,new_confirmed_connections=0)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(source_gate=result,strict_net_source_gain=passed,gains=gains,losses=losses)))
if __name__=='__main__':main()
