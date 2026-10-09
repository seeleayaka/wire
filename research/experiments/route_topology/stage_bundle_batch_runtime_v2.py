"""New immutable protocol after preflight repair; same-turn fresh preparation."""
import json
import os
from pathlib import Path
import subprocess
from bundle_batch_contract import validate_preparation
from bundle_runtime_pins import source_pins
from core import sha256
from run_prompt_contrast import verify
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    original=ROOT/'artifacts/mendeley_confirmed_bundle_batch_20261006'
    protocol=json.loads((original/'protocol.json').read_text(encoding='utf-8'))
    preparation=json.loads((original/'preparation_report.json').read_text(encoding='utf-8'))
    if (original/'progress.json').exists():raise ValueError('v1 inference already started; do not duplicate')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    validate_preparation(preparation['cases'],protocol['cases'])
    folder=Path(__file__).parent
    code="import sys; sys.path.insert(0,"+repr(str(folder))+"); from bundle_runtime_pins import source_pins; print(len(source_pins()))"
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
    probe=subprocess.run(['E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe','-B','-c',code],
                         env=env,capture_output=True,text=True,check=True)
    if probe.stdout.strip()!='8':raise ValueError('actual SAM runtime helper preflight failed')
    output=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    output.mkdir(exist_ok=False)
    save(output/'preparation_report.json',preparation)
    protocol['pins'].update({str(p):sha256(p) for p in [Path(__file__),folder/'bundle_runtime_pins.py',
        folder/'run_bundle_batch_geometry_v2.py',folder/'analyze_confirmed_bundle_batch_v2.py',
        folder/'audit_confirmed_bundle_batch_v2.py',folder/'run_confirmed_bundle_batch_pipeline.py',
        output/'preparation_report.json']})
    protocol.update(runtime_preflight_status='PASS',runtime_fingerprint_helper='stdlib_only_same_E_hashes',
        predecessor_unlaunched_protocol=str(original/'protocol.json'),
        preparation_provenance='same-turn31-original fresh preparation; only repaired runtime import, no SAM or prior inspection intermediate reuse',
        active_runner=str(folder/'run_bundle_batch_geometry_v2.py'))
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift during stage')
    save(output/'protocol.json',protocol)
    save(output/'runtime_preflight.json',{'status':'PASS','SAM_environment_import_verified':True,
        'canonical_E_hash_equality_test_passed':True,'new_training_or_threshold_change':False,
        'v1_SAM_inference_started':False,'fresh_prepared_originals':len(preparation['cases']),
        'planned_fresh_encoders':len(protocol['cases']),'planned_fresh_decoders':len(protocol['cases'])*2})
    print('PASS: new v2 protocol;31 same-turn fresh originals;24 planned new SAM encoders')


if __name__=='__main__':main()
