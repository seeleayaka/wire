"""Explicit, annotation-assisted terminal body inventory; no SAM or graph verdict."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inspection_agent.terminal_annotation_inventory import import_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = import_inventory(args.image, args.annotations)
    # Do not replace existing reports or source annotations.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["counts"],
                      "topology_assessment": result["topology_assessment"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
