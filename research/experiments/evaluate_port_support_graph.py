"""Freeze the checkpoint-support graph, keep the original strict cutoffs."""
from pathlib import Path
import evaluate_strong_student_feature_consensus as evaluator
from port_support_graph_policy import append_support_graph,potential_peer_support,POLICY
from inspection_agent.optional_port_crop_review import sha


def main():
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_support_graph_policy.py'),Path(evaluator.__file__))}
    old=(evaluator.OUT,evaluator.POLICY,evaluator.append_strong_consensus,evaluator.consistent_rows,evaluator.save)
    def save(path,value):
        if path.name=='protocol.json':
            value=dict(value);value['pins']=dict(value['pins'],**pins)
            value.update(policy=POLICY,eligibility_callback_replaced=True,
                exact_short_circuit='No valid first-student raw confidence>.25 or full current budget makes missing peer inference unnecessary',
                strong_score_experiment_rejected_before_this_test=True,
                thresholds_inherited_not_fit_to_result=True,only_new_dimension='checkpoint support graph topology')
        assert {p:sha(Path(p)) for p in pins}==pins
        return old[4](path,value)
    try:
        evaluator.OUT=evaluator.ROOT/'artifacts/port_support_graph_20261003'
        evaluator.POLICY=POLICY;evaluator.append_strong_consensus=append_support_graph
        evaluator.consistent_rows=potential_peer_support;evaluator.save=save;evaluator.main()
    finally:
        evaluator.OUT,evaluator.POLICY,evaluator.append_strong_consensus,evaluator.consistent_rows,evaluator.save=old


if __name__=='__main__':main()
