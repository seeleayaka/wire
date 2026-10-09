"""Bind robust classification to this turn's fresh SAM without claiming new SAM."""
import json
from pathlib import Path
import numpy as np
from core import sha256
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    pair=ROOT/'artifacts/mendeley_compound_bundle_pair_20261006'
    observer=ROOT/'artifacts/mendeley_translation_observer_fresh_20261006'
    prep=json.loads((pair/'preparation_report.json').read_text(encoding='utf-8'))
    inf=json.loads((pair/'inference_report.json').read_text(encoding='utf-8'))
    new=json.loads((observer/'report.json').read_text(encoding='utf-8'))
    proto=json.loads((observer/'protocol.json').read_text(encoding='utf-8'))
    assert new['status']=='complete' and new['source_controls_passed'] and new['prior_supported_loss']==[]
    assert inf['status']=='complete' and inf['protocol_sha256']==sha256(pair/'protocol.json')
    assert source_pins()==proto['mainline_pins'] and all(sha256(p)==d for p,d in proto['pins'].items())
    results=[]
    for r in prep['cases']:
        candidates=[n for n in new['cases'] if n['id']==r['id'] and (n['dataset']=='source_controls' if r['id']=='reference' else n['dataset']=='demo30')]
        assert len(candidates)==1
        n=candidates[0];assert n['origin']['image_sha256']==r['original_source']['image_sha256']
        oldpose=next(a for a in r['anchors'] if a['id']=='FAN_CPU')
        exact=np.array_equal(np.asarray(oldpose['inspection_to_reference_local']),np.asarray(n['pose']['inspection_to_reference_local']))
        same=n['prediction']['phenotype']==r['phenotype']
        assert exact and same
        results.append(dict(id=r['id'],source_sha256=r['original_source']['image_sha256'],CPU_pose_exact_match=exact,
            phenotype_match=same,robust_contact_positive=n['prediction']['positive_contact_cue'],
            robust_contact_scores=n['prediction']['contact_outer_scores']))
    out=ROOT/'artifacts/mendeley_translation_pair_binding_audit_20261006';out.mkdir(exist_ok=False)
    report=dict(status='PASS',cases=results,fresh_SAM_encoders_this_turn=inf['fresh_encoders'],
        no_second_SAM_run=True,reanalysis_binding_only_not_changed_relation_gate=True,
        final_topology_decision='insufficient_evidence',reason='exposed_socket_and_high_score_mask_conflict',
        electrical_connections_confirmed=0,mainline_unchanged=True,deployed=False)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
