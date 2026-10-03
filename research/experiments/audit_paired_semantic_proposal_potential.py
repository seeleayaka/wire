"""Before costly paired training, verify fixed weak geometry has room to help."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,load,save,read_targets
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.port_tiling import box_iou
from port_semantic_model_vote import proposals,PROPOSAL_FLOOR
from audit_port_multiscale_acceptance import metric
from audit_port_matching_cardinality import maximum_matching
OUT=ROOT/'artifacts/paired_semantic_proposal_potential_20261003'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections'


def main():
    if OUT.exists():raise FileExistsError('Preserve proposal audit')
    OUT.mkdir();pins={str(Path(__file__)):sha(Path(__file__)),str(Path(__file__).with_name('port_semantic_model_vote.py')):sha(Path(__file__).with_name('port_semantic_model_vote.py'))}
    frozen=resolution_runtime_fingerprint(REPO);summary={};allrows=[]
    for stage in ('train','inner','outer'):
        totals=dict(current_tp=0,unbounded_candidate_maximum_tp=0,budget_limited_upper_tp=0,targets=0,new_proposals=0,proposal_sources=0)
        for row in load(PREP/stage/'index.json')['records']:
            path=Path(row['path']);assert sha(path)==row['sha256'];pins[str(path)]=row['sha256'];case=load(path);current=case['current'];old=current['all_predictions']
            remaining=5-(len(old)-len(current['primary']))
            candidates=proposals(case['teacher'],[case['teacher'],case['student']]) if remaining else []
            candidates=[p for p in candidates if not any(box_iou(p['box_xyxy'],o['box_xyxy'])>=.5 for o in old)]
            # Finalized candidates are saved BEFORE target boxes are opened.
            save(OUT/(stage+'_'+Path(row['image']).stem+'_proposals.json'),dict(image=row['image'],current=current,candidates=candidates,remaining=remaining))
            targets=read_targets(stage,row['image'],case['teacher']['predictions']['source_shape'],case['entry']['label_sha256'],pins)
            m=metric(old,targets);assert len(maximum_matching(old,targets))==m['tp']
            upper=len(maximum_matching(old+candidates,targets));limited=min(upper,m['tp']+remaining)
            values=dict(current_tp=m['tp'],unbounded_candidate_maximum_tp=upper,budget_limited_upper_tp=limited,targets=len(targets),
                new_proposals=len(candidates),proposal_sources=int(bool(candidates)))
            for k,v in values.items():totals[k]+=v
            allrows.append(dict(stage=stage,image=row['image'],**values))
        summary[stage]=dict(totals,possible_gain=totals['budget_limited_upper_tp']-totals['current_tp'])
    assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
    result=dict(status='complete',summary=summary,cases=allrows,pins=pins,proposal_floor=PROPOSAL_FLOOR,
        existing_semantic_proposal_floor_not_tuned=True,paired_classifier_not_trained_or_tested=True,
        no_candidate_from_placeholder_peer=True,oracle_never_selection_feature=True,field_accuracy=False)
    save(OUT/'report.json',result);print(str(summary),flush=True)


if __name__=='__main__':main()
