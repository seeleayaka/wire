"""Read-only gain cards for full TRAIN result, not photo-specific tuning."""
from pathlib import Path
from prepare_paired_port_semantics import ROOT,DATA,load
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import render_card
SOURCE=ROOT/'artifacts/raw_pose_consensus_20261004/train'
OUT=ROOT/'artifacts/raw_pose_consensus_train_cards_20261004'

def main():
    if OUT.exists():raise FileExistsError('Preserve reviewed cards')
    OUT.mkdir();entries={r['image']:r for r in load(BASE/'train/report.json')['cases']};pins={}
    for row in load(SOURCE/'report.json')['cases']:
        if not row['gained']:continue
        name=row['image'];case=load(SOURCE/(Path(name).stem+'_predictions.json'))
        targets=read_targets('train',name,[2736,3648],entries[name]['label_sha256'],pins)
        for target in row['gained']:
            path=OUT/(Path(name).stem+'_target'+str(target)+'.jpg')
            render_card(DATA/'images/train01'/name,targets,case['current']['all_predictions'],case['trial']['all_predictions'],targets[target]['box'],
                'Accepted native baseline vs raw seed pose; TRAIN development only',path)
            print(path)

if __name__=='__main__':main()
