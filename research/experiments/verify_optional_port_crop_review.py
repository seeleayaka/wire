"""Replay all fixed cases and exercise guarded fallback without annotations."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

from optional_port_crop_review import optional_port_crop_review, load, sha, SCENE, REFERENCE_SHA


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def make_context(prior, source_sha):
    return {"parents":copy.deepcopy(prior["parents"]),"existing_hints":copy.deepcopy(prior["hints"]),
            "source_sha256":source_sha,"reference_sha256":REFERENCE_SHA,
            "source_to_reference_homography":prior["actual_homography"],
            "alignment_reliable":prior["alignment"]["alignment_quality"]["reliable"]}


def main():
    global optional_port_crop_review
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--acceptance",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--installed",action="store_true")
    args=parser.parse_args()
    sys.path.insert(0,str(args.project))
    module_path=Path(__file__).with_name("optional_port_crop_review.py")
    if args.installed:
        from inspection_agent.optional_port_crop_review import optional_port_crop_review as installed_review
        optional_port_crop_review=installed_review
        module_path=args.project / "inspection_agent/optional_port_crop_review.py"
    data=args.project / "data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images"
    reference=data / "train01/normal_073.JPG"
    calibration=args.acceptance / "calibration.json"
    old=load(args.project / "output/port_state_hints_validation_20260929/report.json")
    expected={c["image"]:c for c in load(args.acceptance / "partial_cases.json")}
    cases=[]
    for prior in sorted(old["cases"],key=lambda c:c["image"]):
        name=prior["image"]
        source=data / "val01" / name
        cached=load(args.acceptance / "cache" / ("val01_"+Path(name).stem+".json"))
        context=make_context(prior,cached["source_sha256"])
        save(args.output / "contexts" / (Path(name).stem+".json"),context)
        untouched=copy.deepcopy(context)
        result=optional_port_crop_review(source,reference,context,project=args.project,
            calibration=calibration,enabled=True,scene=SCENE,prediction_provider=lambda image,c=cached:copy.deepcopy(c))
        assert result["status"] == "applied",result["fallback_reason"]
        assert context == untouched
        exp=expected[name]
        for key in ("parents","existing_hints","tile_hints","selection_audit","aligned_predictions"):
            assert result[key] == exp[key],(name,key)
        save(args.output / "replay" / (Path(name).stem+".json"),result)
        cases.append({"image":name,"new_hints":len(result["tile_hints"]),"exact_match":True})

    name="disconnected_002.JPG"
    prior=next(c for c in old["cases"] if c["image"]==name)
    source=data / "val01" / name
    cached=load(args.acceptance / "cache/val01_disconnected_002.json")
    context=make_context(prior,cached["source_sha256"])
    gates=[]
    calls=[]
    def unexpected_provider(image):
        calls.append(1)
        raise RuntimeError("injected_prediction_failure")
    base={"project":args.project,"calibration":calibration,"enabled":True,"scene":SCENE,
          "prediction_provider":unexpected_provider}
    def check(label,ctx=None,source_path=None,ref_path=None,**kwargs):
        options={**base,**kwargs}
        current=copy.deepcopy(ctx or context)
        original=copy.deepcopy(current)
        before=len(calls)
        result=optional_port_crop_review(source_path or source,ref_path or reference,current,**options)
        assert current == original
        assert result["parents"] == original["parents"] and result["existing_hints"] == original["existing_hints"]
        assert result["tile_hints"] == [] and result["status"] in ("fallback","disabled")
        if label != "inference_error":
            assert len(calls)==before,label
        gates.append({"case":label,"status":result["status"],"reason":result["fallback_reason"]})
    check("default_disabled",enabled=False,calibration=Path("missing_calibration.json"))
    check("scene_mismatch",scene="cabinet")
    ctx=copy.deepcopy(context);ctx["alignment_reliable"]=False
    check("alignment_unreliable",ctx=ctx)
    ctx=copy.deepcopy(context);ctx["source_sha256"]="0"*64
    check("source_geometry_mismatch",ctx=ctx)
    check("reference_mismatch",ref_path=source)
    ctx=copy.deepcopy(context);ctx["source_to_reference_homography"]=[[0,0,0]]*3
    check("singular_homography",ctx=ctx)
    with tempfile.TemporaryDirectory(prefix="port-entry-gates-") as temp:
        temp=Path(temp)
        damaged_cal=temp / "calibration.json"
        damaged_cal.write_bytes(calibration.read_bytes()+b" ")
        check("calibration_modified",calibration=damaged_cal)
        wrong_weight=temp / "wrong.pt";wrong_weight.write_bytes(b"invalid checkpoint")
        check("weight_mismatch",weight=wrong_weight)
        check("missing_weight",weight=temp / "missing.pt")
    check("inference_error")
    assert len(calls)==1
    assert len(cases)==30 and sum(c["new_hints"] for c in cases)==6
    normal_added=sum(c["new_hints"] for c in cases if c["image"].startswith("normal_"))
    assert normal_added==0
    summary={"status":"verified","fixed_case_count":30,"exact_replays":30,
             "new_hints":6,"new_normal_hints":normal_added,"fallback_cases":gates,
             "source_and_context_unchanged":True,"live_inference_verified":False,
             "module_sha256":sha(module_path),"installed_module":args.installed,"cases":cases}
    save(args.output / "verification.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("cases","fallback_cases")},ensure_ascii=False))


if __name__=="__main__":
    main()
