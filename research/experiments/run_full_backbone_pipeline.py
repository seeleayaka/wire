"""Finite local full training -> gated source evaluation, no deployment."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/port_full_backbone_20261003'
sys.path.insert(0,'E:/PythonProject10')
from inspection_agent.optional_port_crop_review import sha
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    if (BASE/'pipeline_progress.json').exists():raise FileExistsError('Never duplicate local pipeline')
    smoke=load(BASE/'smoke_v2/report.json');assert smoke['status']=='complete' and smoke['changed_early_backbone_tensors']>0
    scripts=[ROOT/'experiments/train_port_full_backbone.py',ROOT/'experiments/evaluate_full_backbone_acceptance.py',Path(__file__)]
    pins={str(p):sha(p) for p in scripts}
    save(BASE/'pipeline_protocol.json',dict(pins=pins,full_epochs=2,freeze=0,fixed_last_checkpoint=True,
        same_audited656_576_crops_192_48_sources=True,after_training='ALL192 train then inner48 then outer30 if gates pass',
        no_automatic_deployment=True,local_machine_must_remain_active=True,field_accuracy=False))
    for phase,cmd in (('training',[sys.executable,'-B',str(scripts[0]),'--mode','full']),
                      ('source_acceptance',[sys.executable,'-B',str(scripts[1])])):
        assert {p:sha(Path(p)) for p in pins}==pins
        save(BASE/'pipeline_progress.json',dict(status='running',phase=phase))
        with (BASE/(phase+'.log')).open('wb') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(BASE/'pipeline_progress.json',dict(status='running',phase=phase,pid=child.pid))
            code=child.wait()
        if code:
            save(BASE/'pipeline_progress.json',dict(status='failed',phase=phase,return_code=code,log=str(BASE/(phase+'.log'))));return
    acceptance=load(BASE/'evaluation/progress.json')
    save(BASE/'pipeline_progress.json',dict(status='complete',acceptance=acceptance,no_automatic_deployment=True))
if __name__=='__main__':main()
