"""Create an image-bound port-map draft or review it with endpoint evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspection_agent.terminal_mapping import create_mapping_draft, review_mapped_topology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["draft", "review"])
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-type")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--endpoints", type=Path)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    if args.mode == "draft":
        if not args.scene_type or args.map or args.endpoints:
            parser.error("draft requires --scene-type; --map/--endpoints belong to review")
        result = create_mapping_draft(args.image, args.scene_type, read(args.inventory) if args.inventory else None)
    else:
        if not args.map or args.inventory or args.scene_type:
            parser.error("review requires --map; --inventory/--scene-type belong to draft")
        result = review_mapped_topology(read(args.map), args.image, read(args.endpoints) if args.endpoints else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "decision": result.get("decision", "draft_only")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
