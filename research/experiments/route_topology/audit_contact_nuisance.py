"""Recompute all outer-offset decisions, including wrong proposals, not just gains."""
import json
from pathlib import Path
from core import sha256
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    folder=ROOT/'artifacts/mendeley_contact_translation_nuisance_20261006'
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    assert source_pins()==protocol['mainline_pins']
    assert all(sha256(p)==d for p,d in protocol['pins'].items())
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'artifacts/mendeley_component_color_source_20261006/report.json').read_text(encoding='utf-8'))['baseline_LOO_results']
    proposals={r['id']:r for r in json.loads((ROOT/'artifacts/mendeley_component_rescue_source_20261006/report.json').read_text(encoding='utf-8'))['cases']}
    features={r['id']:r for r in report['cases']};wrong=[];loss=[];new=[]
    for r in old:
        i=r['id'];previous=r['prediction']['visual_label_candidate'];label=r['visual_label']
        p=proposals[i]['prediction']['visual_label_candidate']
        values=[]
        for j,score in enumerate(features[i]['nuisance_scores']):
            predicted=previous if previous is not None else (0 if p==0 and score>report['threshold'] else None)
            values.append(predicted)
            if predicted is not None and predicted!=label:wrong.append((i,j))
            if previous==label and predicted!=label:loss.append((i,j))
        assert values[4]==features[i]['prediction']['visual_label_candidate']
        if previous is None and values[4]==label:
            assert all(v==label for v in values);new.append(i)
    assert wrong==[] and loss==[] and new==report['gains'] and report['strict_robust_source_gain']
    out=ROOT/'artifacts/mendeley_contact_nuisance_audit_20261006';out.mkdir(exist_ok=False)
    result=dict(status='PASS',checked_decisions=80*9,wrong=wrong,losses=loss,gains=new,
        prior_colour_semantic_and_baseline_heads_fixed_nominal_not_reinferred_at_outer_offsets=True,
        scores_reused_explicitly_not_independent_pixel_feature_replay=True,mainline_unchanged=True,
        not_field_or_topology_acceptance=True,report_sha256=sha256(folder/'report.json'))
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
