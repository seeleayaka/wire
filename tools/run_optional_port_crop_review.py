"""Single-image optional port hint CLI; without --enable, no model is loaded."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--context",type=Path,required=True,
                        help="Label-free JSON with parents, existing_hints, geometry and source/reference hashes")
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--enable",action="store_true")
    parser.add_argument("--scene",default="unknown")
    parser.add_argument("--weight",type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.output.parent / "ultralytics_config")
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    if args.enable:
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.project.resolve()))
    # Installed project module; standalone staging script uses its sibling copy.
    try:
        from inspection_agent.optional_port_crop_review import optional_port_crop_review
    except ModuleNotFoundError as error:
        if error.name != "inspection_agent.optional_port_crop_review":
            raise
        from optional_port_crop_review import optional_port_crop_review
    context = json.loads(args.context.read_text(encoding="utf-8"))
    result = optional_port_crop_review(args.image,args.reference,context,project=args.project,
        calibration=args.calibration,enabled=args.enable,scene=args.scene,weight=args.weight)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"fallback_reason":result["fallback_reason"],
                      "new_hints":len(result["tile_hints"]),"output":str(args.output)},ensure_ascii=False))


if __name__ == "__main__":
    main()
