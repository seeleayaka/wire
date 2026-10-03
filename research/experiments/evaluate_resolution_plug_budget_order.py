"""Frozen270-source evaluator, change class-filter order only; preserve v1."""
from pathlib import Path
import evaluate_port_resolution_plugs as evaluator
from port_resolution_plug_policy_v2 import append_resolution_plugs_v2,POLICY_ID
from inspection_agent.optional_port_crop_review import sha
def main():
    before={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_resolution_plug_policy_v2.py'),Path(evaluator.__file__))}
    oldfunc,oldout,oldid,oldsave=evaluator.append_resolution_plugs,evaluator.OUT,evaluator.POLICY_ID,evaluator.save
    def save(path,value):
        if path.name=='protocol.json':
            value=dict(value);value['pins']=dict(value['pins'],**before)
            value.update(allowed_class_filtered_before_budget=True,selection_callback_replaced=True,
                only_change='Exclude jack rows/rechecks from alternative view before native budget, scores/windows/weights/old cues untouched',
                new_recheck_seeds_not_inferred=True)
        if path.name=='progress.json':assert {p:sha(Path(p)) for p in before}==before
        return oldsave(path,value)
    try:
        evaluator.append_resolution_plugs=append_resolution_plugs_v2
        evaluator.OUT=evaluator.ROOT/'artifacts/resolution_plug_budget_order_20261003'
        evaluator.POLICY_ID=POLICY_ID;evaluator.save=save;evaluator.main()
    finally:evaluator.append_resolution_plugs=oldfunc;evaluator.OUT=oldout;evaluator.POLICY_ID=oldid;evaluator.save=oldsave
if __name__=='__main__':main()
