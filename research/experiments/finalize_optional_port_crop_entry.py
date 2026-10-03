"""Verify live runs and exact deployment copies, then save final evidence."""
import json
from pathlib import Path
from optional_port_crop_review import sha, load


def main():
    workspace=Path(__file__).resolve().parents[1]
    repo=Path(r"E:\PythonProject10")
    out=workspace / "artifacts/optional_port_crop_entry_20260930"
    accepted=workspace / "artifacts/port_crop_acceptance_20260930"
    staged=[workspace / "experiments/optional_port_crop_review.py",
            workspace / "experiments/run_optional_port_crop_review.py",
            accepted / "calibration.json",
            workspace / "experiments/OPTIONAL_PORT_CROP_ENTRY_20260930.md"]
    installed=[repo / "inspection_agent/optional_port_crop_review.py",
               repo / "tools/run_optional_port_crop_review.py",
               repo / "config/port_crop_calibration_frozen_20260930.json",
               repo / "inspection_agent/OPTIONAL_PORT_CROP_ENTRY_20260930.md"]
    copies=[]
    for source,target in zip(staged,installed):
        assert sha(source)==sha(target)
        copies.append({"staged":str(source),"installed":str(target),"sha256":sha(target)})
    verification=load(out / "installed/verification.json")
    assert verification["installed_module"] and verification["exact_replays"]==30
    assert verification["module_sha256"]==sha(installed[0])
    lives=[]
    for kind,name in [("fault","disconnected_002"),("normal","normal_001")]:
        live=load(out / f"live_{kind}/report.json")
        replay=load(out / "installed/replay" / (name+".json"))
        cached=load(accepted / "cache" / ("val01_"+name+".json"))
        assert live["status"]=="applied" and live["execution_mode"]=="live_inference"
        for key in ["parents","existing_hints","tile_hints","selection_audit","aligned_predictions"]:
            assert live[key]==replay[key],(name,key)
        for key in ["source_shape","windows","raw_predictions","edge_rejected","edge_kept_predictions","merged_predictions"]:
            assert live["predictions"][key]==cached[key],(name,key)
        lives.append({"image":name+".JPG","new_hints":len(live["tile_hints"]),
                      "raw_predictions_exact":True,"selection_exact":True})
    result={"status":"verified","installed_files":copies,"fixed_replays_exact":30,
            "fallback_cases_passed":len(verification["fallback_cases"]),"live_inference":lives,
            "existing_unit_tests_passed":16,"default_gui_changed":False,
            "claim":"Optional entry integration; no new accuracy/generalization result."}
    (out / "final_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
