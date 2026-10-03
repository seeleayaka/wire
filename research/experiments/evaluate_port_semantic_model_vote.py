"""Reuse frozen OOF evaluator with a separately pinned proposal-only channel.

The old evaluator file is retained byte-identical. Only its proposal callback
and output destination are replaced; protocol explicitly records the effective
new rule and pins this wrapper/policy. Actual images, heads, crop inference,
all168 cases, metrics and acceptance gates are unchanged.
"""
import json
from pathlib import Path
import evaluate_port_semantic_support as evaluator
from port_semantic_model_vote import proposals,PROPOSAL_FLOOR
from inspection_agent.optional_port_crop_review import sha

OUT=evaluator.ROOT/'artifacts/port_semantic_model_vote_20261003'
def main():
    original=evaluator.load(evaluator.ROOT/'artifacts/port_semantic_source_support_20261003/report.json')
    assert original['status']=='complete' and not original['qualifies']
    before={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_semantic_model_vote.py'),
        Path(evaluator.__file__))}
    original_proposals,original_out,original_save=evaluator.proposals,evaluator.OUT,evaluator.save
    def save(path,value):
        if path.name=='protocol.json':
            value=dict(value)
            value['pins']=dict(value['pins'],**before)
            value.pop('teacher_sameclass_support_above_floor',None)
            value.pop('two_complete_distinct_raw_tiles_above_floor',None)
            value.update(new_candidate_detector_floor=PROPOSAL_FLOOR,
                effective_proposal_policy='At least2 DISTINCT checkpoint SHAs agree same class IoU>=.5 at>.05; complete16px box; semantic>=.98 OOF; old5+5 cues retained',
                same_weight_different_resolutions_not_independent_votes=True,
                proposal_only_callback_replaced=True,original_evaluator_sha256=before[str(Path(evaluator.__file__))],
                old_deployed_thresholds_unchanged=True,new_channel_is_experimental_lower_score_proposals=True,
                low_proposal_scores_not_fault_confidence=True,no_semantic_threshold_reduction=True)
        if path.name in ('report.json','progress.json') and value.get('status')=='complete':
            assert {p:sha(Path(p)) for p in before}==before
        return original_save(path,value)
    try:
        evaluator.proposals=proposals;evaluator.OUT=OUT;evaluator.save=save
        evaluator.main()
    finally:evaluator.proposals=original_proposals;evaluator.OUT=original_out;evaluator.save=original_save
if __name__=='__main__':main()
