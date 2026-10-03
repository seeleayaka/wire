"""V3 native rankings unchanged, class gate before final budget,270-source audit."""
from pathlib import Path
import evaluate_port_resolution_plugs as evaluator
from port_resolution_plug_policy_v3 import append_resolution_plugs_v3,POLICY_ID
from inspection_agent.optional_port_crop_review import sha
def main():
    before={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_resolution_plug_policy_v3.py'),Path(evaluator.__file__))}
    oldfunc,oldout,oldid,oldsave=evaluator.append_resolution_plugs,evaluator.OUT,evaluator.POLICY_ID,evaluator.save
    def save(path,value):
        if path.name=='protocol.json':
            value=dict(value);value['pins']=dict(value['pins'],**before)
            value.update(allowed_class_filtered_before_final_budget=True,native_model_selector_unchanged=True,
                only_change='Skip jack candidate additions before final shared-extra-slot allocation;do not change native primary/supplemental rankings',
                selection_callback_replaced=True)
        if path.name=='progress.json':assert {p:sha(Path(p)) for p in before}==before
        return oldsave(path,value)
    try:
        evaluator.append_resolution_plugs=append_resolution_plugs_v3
        evaluator.OUT=evaluator.ROOT/'artifacts/resolution_plug_final_budget_20261003'
        evaluator.POLICY_ID=POLICY_ID;evaluator.save=save;evaluator.main()
    finally:evaluator.append_resolution_plugs=oldfunc;evaluator.OUT=oldout;evaluator.POLICY_ID=oldid;evaluator.save=oldsave
if __name__=='__main__':main()
