"""Identify which original label line each new precise result matched."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(r"E:\PythonProject10")
ROOT = Path(__file__).resolve().parents[1] / "artifacts/port_crop_acceptance_20260930"
sys.path.insert(0,str(REPO))
from tools.evaluate_mendeley_local_refinement import iou


def load(path):return json.loads(path.read_text(encoding="utf-8"))


def main():
    cases=load(ROOT / "partial_cases.json")
    lines=[]
    for case in cases:
        if not case["tile_hints"]:continue
        old_boxes=case["parents"]+[h["box"] for h in case["existing_hints"]]
        hint=case["tile_hints"][0]["box"]
        scored=[]
        for index,(target,cls) in enumerate(zip(case["targets"],case["source_classes"]),1):
            old_best=max((iou(b,target) for b in old_boxes),default=0.)
            hit=iou(hint,target)
            scored.append({"label_line":index,"source_class":cls,"new_hint_iou":hit,"old_best_iou":old_best})
        scored.sort(key=lambda r:r["new_hint_iou"],reverse=True)
        best=scored[0]
        lines.append({"image":case["image"],"hint_source_class":hint["class_id"]+3,
                      "best_label_line":best["label_line"],"best_label_class":best["source_class"],
                      "best_iou":round(best["new_hint_iou"],4),"old_best_iou":round(best["old_best_iou"],4),
                      "new_precise_match":best["new_hint_iou"]>=.5 and best["source_class"]==hint["class_id"]+3})
    assert len(lines)==6 and sum(r["new_precise_match"] for r in lines)==5
    target=ROOT / "new_hint_original_label_matches.json"
    assert not target.exists()
    target.write_text(json.dumps(lines,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(lines,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
