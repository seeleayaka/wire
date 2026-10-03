"""Print fixed release provenance for apply_patch; never write core files."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_semantic_geometry import NEW
from prepare_paired_port_semantics import ROOT,REPO,load,sha
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint


def main():
    trained=load(NEW/'head_full/report.json');live_path=ROOT/'artifacts/paired_geometry_live_resume_20261003/report.json';live=load(live_path)
    parity_path=ROOT/'artifacts/paired_geometry_formalization_parity_20261003/report.json';parity=load(parity_path)
    assert live['qualifies_live_diagnostic'] and parity['all270_proposal_and_selector_dictionary_parity']
    assert trained['no_validation_training'] and trained['runtime_fingerprint']==resolution_runtime_fingerprint(REPO)
    summaries={};pins={}
    for stage in ('full_train','holdouts/inner','holdouts/outer'):
        path=NEW/stage/'report.json';report=load(path);pins[str(path)]=sha(path);summaries[stage]=report['summary']
        assert report['status']=='complete'
        assert report['summary']['trial']['unmatched']<=report['summary']['current']['unmatched']
        assert all(not r['lost'] for r in report['cases'])
    assert summaries['full_train']['trial']['tp']>summaries['full_train']['current']['tp']
    assert summaries['holdouts/inner']['trial']['tp']>summaries['holdouts/inner']['current']['tp']
    helper=ROOT/'staging/paired_geometry_release/paired_port_features.py'
    for path in (NEW/'head_full/report.json',live_path,parity_path,ROOT/'artifacts/paired_geometry_release_preregistration_20261003/PLAN.md'):
        pins[str(path)]=sha(path)
    manifest=dict(policy_id='current_V3_preserved_paired_geometry_v1_20261003',head_sha256=trained['checkpoint']['sha256'],
        encoder_sha256='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9',
        features_sha256=sha(helper),accepted_runtime=trained['runtime_fingerprint'],probability_gate=.98,
        feature_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],source_gates_passed=True,
        live_diagnostic_passed=True,no_validation_training=True,source_summaries=summaries,live_gains=live['gains'],
        evidence_pins=pins,default_enabled=False,primary_budget=5,supplementary_budget=5,
        reference_minimum_probability=.25,reference_iou=.5,context_minimum_valid_fraction=.85,
        automatic_fault_verdict=False,independent_field_accuracy=False,validation_reused=True,
        live_safety_abstentions=live['safety_abstentions'],sam_complete_acceptance_pending=True)
    assert sha(REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth')==manifest['encoder_sha256']
    print(json.dumps(manifest,indent=2)+'\n',end='')

if __name__=='__main__':main()
