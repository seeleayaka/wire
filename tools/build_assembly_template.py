"""Build an automatic single-reference assembly template and audit artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assembly_engine.assembly_template import save_template  # noqa: E402
from assembly_engine.reference_builder import build_reference_template, read_image, write_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Approved real cabinet photo.")
    parser.add_argument("--template", type=Path, default=ROOT / "config" / "current_cabinet_assembly_template.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "assembly_template_build")
    parser.add_argument("--template-id", default="current_cable_cabinet_v1")
    args = parser.parse_args()
    reference = read_image(args.reference)
    template, overlay, report = build_reference_template(reference, str(args.reference), args.template_id)
    save_template(template, args.template)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output / "reference_candidates.jpg", overlay)
    (args.output / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"template": str(args.template), "output": str(args.output), **template.summary()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
