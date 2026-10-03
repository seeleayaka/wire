"""Finite source-disjoint conditional semantics with layered qualification."""
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,OUT,load,save,sha


def main():
    if (OUT/'pipeline_progress.json').exists():raise FileExistsError('No duplicate pipeline')
    smoke=load(OUT/'smoke/verification.json');assert smoke['status']=='complete' and smoke['finite_gradient_step'] and smoke['original_registration_cache_parity']
    scripts=[ROOT/'experiments'/name for name in ('prepare_paired_port_semantics.py','train_paired_port_semantic_heads.py',
        'evaluate_paired_port_semantic_train.py','train_paired_port_semantic_full.py','evaluate_paired_port_semantic_holdouts.py')]
    paths=scripts+[Path(__file__),ROOT/'experiments/paired_port_semantics.py',ROOT/'experiments/paired_port_semantic_selection.py',
        ROOT/'experiments/port_semantic_verifier.py',ROOT/'experiments/current_port_baseline_audit.py',ROOT/'experiments/audit_port_multiscale_acceptance.py',
        ROOT/'artifacts/paired_port_semantics_preregistration_20261003/PLAN.md']
    pins={str(p):sha(p) for p in paths};save(OUT/'pipeline_protocol.json',dict(pins=pins,stages=['train192_pair_features','classifier_OOF_crop_feasibility',
        'ALL192_actual_OOF_proposal_source_gate','fixed_full_head_train_only','inner48_then_outer30'],
        smoke_gradient_and_registration_cache_parity_verified=True,OOF_only_classifier_not_original_YOLO=True,
        no_threshold_or_label_changes=True,no_automatic_deployment=True,local_machine_required=True,field_accuracy=False))
    for index,(phase,args) in enumerate((('train192_pair_features',['--mode','full']),('classifier_oof',[]),('source_train_oof',[]),('full_head',[]),('holdouts',[]))):
        assert {p:sha(Path(p)) for p in pins}==pins
        with (OUT/(phase+'.log')).open('wb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(scripts[index]),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(OUT/'pipeline_progress.json',dict(status='running',phase=phase,pid=child.pid));code=child.wait()
        if code:save(OUT/'pipeline_progress.json',dict(status='failed',phase=phase,return_code=code));return
        if phase=='classifier_oof' and not load(OUT/'heads_oof/report.json')['qualifies_crop_feasibility']:
            save(OUT/'pipeline_progress.json',dict(status='rejected_crop_feasibility',summary=load(OUT/'heads_oof/report.json')['summary'],actual_source_gate_not_run=True));return
        if phase=='source_train_oof' and not load(OUT/'source_train_oof/report.json')['qualifies']:
            save(OUT/'pipeline_progress.json',dict(status='rejected_OOF_source_train',summary=load(OUT/'source_train_oof/report.json')['summary'],full_head_and_holdouts_not_run=True));return
    save(OUT/'pipeline_progress.json',dict(status='complete',acceptance=load(OUT/'holdouts/progress.json'),no_automatic_deployment=True))


if __name__=='__main__':main()
