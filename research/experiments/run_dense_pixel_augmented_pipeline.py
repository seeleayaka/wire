"""Finite fixed augmentation training then unchanged crop/source gates."""
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from train_dense_pixel_augmented import ROOT,BASE,load,save,sha


def main():
    if (BASE/'pipeline_progress.json').exists():raise FileExistsError('No duplicate pipeline')
    assert load(BASE/'smoke/report.json')['status']=='complete'
    scripts=[ROOT/'experiments/train_dense_pixel_augmented.py',ROOT/'experiments/evaluate_dense_pixel_augmented_source.py']
    paths=scripts+[Path(__file__),ROOT/'experiments/dense_pixel_geometric_augmentation.py',ROOT/'experiments/dense_pixel_port_probe.py',
        ROOT/'experiments/dense_port_probe.py',ROOT/'experiments/evaluate_dense_dino_source_support.py',ROOT/'experiments/current_port_baseline_audit.py',
        ROOT/'artifacts/dense_pixel_augmented_preregistration_20261003/PLAN.md']
    pins={str(p):sha(p) for p in paths}
    save(BASE/'pipeline_protocol.json',dict(pins=pins,fixed_additional16_total24_epochs=True,training_augmentation_only=True,
        same_scores_and_budget=True,crop_precision90_recall25_required=True,no_automatic_deployment=True,field_accuracy=False))
    for phase,index,args in (('augmented_training',0,['--mode','full']),('source_acceptance',1,[])):
        assert {p:sha(Path(p)) for p in pins}==pins
        with (BASE/(phase+'.log')).open('wb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(scripts[index]),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(BASE/'pipeline_progress.json',dict(status='running',phase=phase,pid=child.pid));code=child.wait()
        if code:save(BASE/'pipeline_progress.json',dict(status='failed',phase=phase,return_code=code));return
        if phase=='augmented_training' and not load(BASE/'full/report.json')['qualifies_crop_feasibility']:
            save(BASE/'pipeline_progress.json',dict(status='rejected_crop_feasibility',crop_feasibility=load(BASE/'full/report.json')['crop_feasibility'],no_source_accuracy_evaluated=True));return
    save(BASE/'pipeline_progress.json',dict(status='complete',acceptance=load(BASE/'source_acceptance/progress.json'),no_automatic_deployment=True))


if __name__=='__main__':main()
