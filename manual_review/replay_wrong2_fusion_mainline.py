"""Run the formal DINO GUI mainline headlessly and wait for SAM3 fusion.

This is a repeatable regression harness for the cabinet-4 wrong2 case.  It
uses the same DINOReview class launched by the normal batch file, rather than
calling an experimental fusion script directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_CASE / "right.png")
    parser.add_argument("--inspection", type=Path, default=DEFAULT_CASE / "wrong2.png")
    parser.add_argument("--progress-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.reference.is_file() or not args.inspection.is_file():
        raise FileNotFoundError("Both --reference and --inspection must be existing images")
    # The formal entry must import Torch before a Qt import on this Windows host.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(PROJECT / "prototype"))
    import assembly_auto_review_dino_v2 as entry  # noqa: PLC0415
    from PyQt5.QtWidgets import QApplication  # noqa: PLC0415

    application = QApplication([])
    window = entry.DINOReview()
    window.reference.setText(str(args.reference))
    window.inspection.setText(str(args.inspection))
    window.recipe["check_rois"] = [[0.01, 0.02, 0.99, 0.98]]
    window.run()
    if window.current_output is None:
        raise RuntimeError("The formal mainline did not create an output directory")
    pending = window._sam3_pending
    print(json.dumps({
        "stage": "dino_complete_sam3_started" if pending else "completed_without_sam3_worker",
        "output": str(window.current_output),
        "dino_candidates": len(pending["dino_regions"]) if pending else None,
    }, ensure_ascii=False), flush=True)

    started = time.monotonic()
    last_progress = 0
    while window.sam3_worker is not None:
        application.processEvents()
        elapsed = int(time.monotonic() - started)
        if elapsed - last_progress >= args.progress_seconds:
            print(json.dumps({"stage": "sam3_running", "elapsed_seconds": elapsed}, ensure_ascii=False), flush=True)
            last_progress = elapsed
        time.sleep(0.1)
    application.processEvents()
    report = json.loads((window.current_output / "report.json").read_text(encoding="utf-8"))
    fusion = report.get("sam3_fusion", {})
    print(json.dumps({
        "stage": "complete",
        "output": str(window.current_output),
        "decision": report.get("decision"),
        "sam3_status": fusion.get("status"),
        "green_boxes": [item.get("bbox_xyxy") for item in fusion.get("green_regions", [])],
        "yellow_boxes": [item.get("bbox_xyxy") for item in fusion.get("yellow_review_regions", [])],
    }, ensure_ascii=False), flush=True)
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
