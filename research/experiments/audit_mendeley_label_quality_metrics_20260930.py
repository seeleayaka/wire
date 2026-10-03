"""Read-only label-subset metrics; never substitutes for semantic review."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

REPO=Path(r"E:\PythonProject10")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from tools.evaluate_mendeley_local_refinement import iou


def load(path):return json.loads(path.read_text(encoding="utf-8"))


def main():
    cases=load(ROOT/"artifacts/port_crop_acceptance_20260930/partial_cases.json")
    geometry=load(ROOT/"artifacts/mendeley_val_label_audit_20260930_v2/geometry_report.json")
    by_class={str(c):Counter() for c in (1,2,3,4)}
    for case in cases:
        base=case["parents"]+[h["box"] for h in case["existing_hints"]]
        proposed=base+[h["box"] for h in case["tile_hints"]]
        for cls,target in zip(case["source_classes"],case["targets"]):
            old=max((iou(b,target) for b in base),default=0.)
            new=max((iou(b,target) for b in proposed),default=0.)
            group=by_class[str(cls)]
            group["labels"]+=1
            group["old_iou_ge_05"]+=old>=.5
            group["new_iou_ge_05"]+=new>=.5
    assert sum(v["labels"] for v in by_class.values())==245
    assert sum(v["old_iou_ge_05"] for v in by_class.values())==5
    assert sum(v["new_iou_ge_05"] for v in by_class.values())==10
    tiny=[r for r in geometry["records"] if min(r["box"][2]-r["box"][0],r["box"][3]-r["box"][1])<15]
    result={"by_source_class":{k:dict(v) for k,v in by_class.items()},"short_side_under15px_count":len(tiny),
            "short_side_under15px_examples":[{"image":r["image"],"label_line":r["index"],"class":r["class"]} for r in tiny[:12]],
            "warning":"Class-specific IoU uses existing candidate boxes, not automatic fault classification or expert validation."}
    target=ROOT/"artifacts/mendeley_val_label_audit_20260930_v2/label_subset_metrics.json"
    assert not target.exists()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
