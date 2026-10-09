"""Separate-process wrapper; old trial files remain hash-verifiable."""
import json
from pathlib import Path
import run_endpoint_pair_expanded as runner
from endpoint_pair_candidate_v3 import compare_endpoint_pair
from run_prompt_contrast import digest,save,verify


def main():
    files=[Path(__file__).resolve(),Path(__file__).with_name('endpoint_pair_candidate_v3.py')]
    pins={str(p):digest(p) for p in files}
    runner.OUT=runner.ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008'
    runner.compare_endpoint_pair=compare_endpoint_pair
    runner.main()
    verify(pins)
    protocol=json.loads((runner.OUT/'protocol.json').read_text(encoding='utf-8'))
    protocol['pins'].update(pins)
    protocol['endpoint_candidate_schema_version']=3
    protocol['single_socket_original_evidence_preserved']=True
    save(runner.OUT/'protocol.json',protocol)
    report=json.loads((runner.OUT/'report.json').read_text(encoding='utf-8'))
    assert all(r['decision']=='reference_socket_exposure_observed' for r in report['cases']
               if r['automatic_topology_comparison']['decision']=='visible_socket_attachment_change_supported')


if __name__=='__main__':main()
