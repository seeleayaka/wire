"""Storage/numeric replay of rejected source trial, not independent recognition."""
import json
from pathlib import Path
import numpy as np
from core import sha256
from component_cues import prediction, gate
from socket_hog_candidate import resolve
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=ROOT/'artifacts/mendeley_hog_socket_source_v2_20261007'
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    raw=json.loads((folder/'model.json').read_text(encoding='utf-8'))
    assert report['protocol_sha256']==sha256(folder/'protocol.json')
    assert all(sha256(p)==v for p,v in protocol['pins'].items())
    assert source_pins()==protocol['mainline_pins']
    model={k:np.asarray(raw[k]) if k in ['center','scale','weights'] else raw[k]
           for k in ['center','scale','weights','bias']}
    calibration={int(k):v for k,v in raw['calibration'].items()}
    inventory={r['id']:r for r in report['prepared']}
    assert len(inventory)==197 and len(set(protocol['fit_ids'])&set(protocol['calibration_ids']))==0
    replay=[];gains=[];losses=[]
    for r in report['cases']:
        source=inventory[r['id']]
        ps=[prediction(model,f,calibration,excluded_id=r['id']) for f in source['probe_features']]
        assert ps==r['shape_probes']
        final=resolve(r['old_candidate'],[p['visual_label_candidate'] for p in ps],r['color_labels'])
        assert final==r['prediction']['visual_label_candidate']
        if r['old_candidate'] is None and final==r['visual_label']:gains.append(r['id'])
        if r['old_candidate'] is not None and final!=r['old_candidate']:losses.append(r['id'])
        replay.append(dict(id=r['id'],visual_label=r['visual_label'],prediction=dict(visual_label_candidate=final)))
    assert gate(replay)==report['source_gate'] and gains==report['source_gains'] and losses==report['source_losses']
    assert report['strict_net_source_gain'] is False and not gains and not losses
    out=ROOT/'artifacts/mendeley_hog_socket_source_replay_20261007';out.mkdir(exist_ok=False)
    save(out/'report.json',dict(status='PASS',source_report_sha256=sha256(folder/'report.json'),
        reconstructed_cases=len(replay),source_prepared=len(inventory),shared_descriptor_and_rank_code=True,
        features_explicitly_reused=True,not_independent_pixel_or_training_audit=True,
        source_gate=report['source_gate'],strict_net_source_gain=False,new_topology_hits=0,
        E_unchanged=True,deployed=False))
    print('PASS80 stored head/rank/rescue/count replay; rejected0gain, E unchanged')


if __name__=='__main__':main()
