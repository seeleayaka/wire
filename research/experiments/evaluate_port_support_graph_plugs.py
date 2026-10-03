"""ALL192 training gate for the existing loose-plug-specific support boundary."""
from pathlib import Path
import evaluate_strong_student_feature_consensus as evaluator
from port_support_graph_plugs_policy import append_graph_plugs,potential_plug_support,POLICY
from inspection_agent.optional_port_crop_review import sha


def main():
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_support_graph_plugs_policy.py'),
          Path(__file__).with_name('port_support_graph_policy.py'),Path(evaluator.__file__))}
    old=(evaluator.OUT,evaluator.POLICY,evaluator.append_strong_consensus,evaluator.consistent_rows,evaluator.save)
    def save(path,value):
        if path.name=='protocol.json':
            value=dict(value);value['pins']=dict(value['pins'],**pins)
            value.update(policy=POLICY,eligibility_callback_replaced=True,
                exact_short_circuit='No valid first-student loose-plug raw confidence>.25 or full current budget makes missing peer inference unnecessary',
                all_class_graph_rejected_before_this_test=True,training_derived_class_boundary=True,
                validation_reused=True,not_new_independent_validation=True,
                thresholds_inherited_not_fit_to_result=True,only_new_dimension='checkpoint support graph topology under current loose-plug-only scope')
        assert {p:sha(Path(p)) for p in pins}==pins
        return old[4](path,value)
    try:
        evaluator.OUT=evaluator.ROOT/'artifacts/port_support_graph_plugs_20261003'
        evaluator.POLICY=POLICY;evaluator.append_strong_consensus=append_graph_plugs
        evaluator.consistent_rows=potential_plug_support;evaluator.save=save;evaluator.main()
    finally:
        evaluator.OUT,evaluator.POLICY,evaluator.append_strong_consensus,evaluator.consistent_rows,evaluator.save=old


if __name__=='__main__':main()
