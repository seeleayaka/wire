"""Read-only model-output recovery if worker final bookkeeping lacks skimage.

Never launches inference or overwrites the worker's failure evidence.
"""
import json
from pathlib import Path
from run_prompt_contrast import save,verify
from run_audit import source_pins
from core import sha256

def main():
    out=Path(__file__).resolve().parents[2]/'artifacts/mendeley_quarter_turn_controls_20261007'
    progress=json.loads((out/'progress.json').read_text(encoding='utf-8'))
    assert progress['status']=='failed' and progress['error']=="No module named 'skimage'",'only explicit final bookkeeping failure recoverable'
    assert not (out/'report.json').exists(),'preserve existing report'
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    rows=[]
    for identity in progress['completed']:
        ap=out/identity/'case_audit.json';audit=json.loads(ap.read_text(encoding='utf-8'));assert audit['status']=='PASS'
        rows.append(dict(id=identity,observation=audit['observation'],audit_sha256=sha256(ap)))
    ids=[r['id'] for r in rows];assert ids==protocol['all_five_planned_ids'][:len(ids)] and ids
    views={r['id']:r['observation'] for r in rows};stop=None
    if not views['reference']['unique_native_bundle_observation_supported']:stop='reference_native_component_failed'
    elif 'source_visible_01' in views and not views['source_visible_01']['unique_native_bundle_observation_supported']:stop='visible01_no_new_native_gain'
    assert stop or len(ids)==5,'no recovery of partial inference'
    gate=len(ids)==5 and all(views[k]['unique_native_bundle_observation_supported'] for k in ['reference','source_visible_01','source_visible_02']) and not any(views[k]['high_score_mask_touches_socket'] for k in ['source_exposed_01','source_exposed_02'])
    save(out/'report.json',dict(status='complete',worker_bookkeeping_failed=True,worker_failure_preserved=True,completed=ids,unexecuted=[k for k in protocol['all_five_planned_ids'] if k not in ids],cases=rows,source_gate_passed=gate,stop_reason=stop,fresh_encoders=len(ids),fresh_decoders=2*len(ids),protocol_sha256=sha256(out/'protocol.json'),visual_review='pending',model_observer_count=1,electrical_connections_confirmed=0,deployed=False))
    print('final bookkeeping recovered from independently audited artifacts; worker failure preserved')

if __name__=='__main__':main()
