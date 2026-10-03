"""Fixed two-view duplicate grouping and unmodified old-ROI distance audit."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from itertools import combinations
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from segment_evidence_groups import group_segment_evidence

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/sam_crop_coverage_20261001"
OUT = ROOT / "artifacts/segment_port_audit_20261001"
OUT.mkdir(parents=True, exist_ok=True)
load = lambda p: json.loads(p.read_text(encoding="utf-8"))
protocol = load(SOURCE / "frozen_protocol.json")
reports = []
for case in protocol["cases"]:
    run = Path(case["fresh_run_directory"])
    raw = load(run / "bound_endpoint_report.json")
    shifted = load(SOURCE / (case["case"] + "_source_endpoints.json"))
    records = deepcopy(raw["records"])
    masks = {}
    cache = {}
    for record, source_record in zip(records, shifted["records"]):
        assert record["record_id"] == source_record["record_id"]
        record["crop_boundary_guard_eligible"] = not source_record["crop_evidence"]["boundary_truncated"]
        mask_id = record["source_mask_id"]
        if mask_id not in cache:
            with Image.open(run / "sam" / f"mask_{mask_id:03d}.png") as image:
                binary = (np.asarray(image.convert("L"))>0).astype(np.uint8)
            cache[mask_id] = cv2.connectedComponents(binary, connectivity=8)[1]
        masks[record["record_id"]] = cache[mask_id] == record["component_id"]
    grouped = group_segment_evidence(records, masks)
    overlap_diagnostics = []
    for first, second in combinations(sorted(masks), 2):
        a, b = masks[first], masks[second]
        overlap = int((a & b).sum())
        if overlap:
            overlap_diagnostics.append(dict(first=first, second=second,
                                           mask_iou=overlap / int((a | b).sum())))
    grouped['largest_overlaps_diagnostic_only'] = sorted(overlap_diagnostics, key=lambda p: (-p['mask_iou'], p['first'], p['second']))[:10]
    reverse = group_segment_evidence(list(reversed(records)), masks)
    reverse['largest_overlaps_diagnostic_only'] = grouped['largest_overlaps_diagnostic_only']
    assert grouped == reverse, "input order changed grouping"
    assert sorted(r for g in grouped["groups"] for r in g["member_record_ids"]) == sorted(r["record_id"] for r in records)
    grouped["source_binding"] = case["source_binding"]
    grouped["crop_binding"] = case["crop_binding"]
    (OUT / (case["case"] + "_groups.json")).write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")
    reports.append({"case": case["case"], **{key:grouped[key] for key in
        ["record_count", "review_group_count", "duplicate_group_count", "display_redundancy_reduction"]}})
    # Preserve raw evidence. Representative sheet is an optional review presentation.
    columns, tw, th = 6, 180, 150
    sheet = Image.new("RGB",(columns*tw,((len(grouped["groups"])+columns-1)//columns)*th),"#f2f4f7")
    draw = ImageDraw.Draw(sheet)
    with Image.open(case["crop_binding"]["image_path"]) as image:
        base = image.convert("RGB")
    for i, group in enumerate(grouped["groups"]):
        mask = masks[group["representative_record_id"]]
        ys,xs = np.where(mask)
        left,top=max(0,int(xs.min())-8),max(0,int(ys.min())-8)
        right,bottom=min(base.width,int(xs.max())+9),min(base.height,int(ys.max())+9)
        pixels=np.asarray(base.crop((left,top,right,bottom))).astype(np.float32)
        active=mask[top:bottom,left:right]
        pixels[active]=pixels[active]*.5+np.array([40,220,180])*.5
        tile=Image.fromarray(pixels.astype(np.uint8));tile.thumbnail((tw-10,th-40))
        x,y=(i%columns)*tw,(i//columns)*th
        draw.text((x+4,y+4),f"G{i+1:02} {group['representative_record_id']} n={len(group['member_record_ids'])}",fill="black")
        sheet.paste(tile,(x+(tw-tile.width)//2,y+30))
    sheet.save(OUT / (case["case"] + "_review_groups.png"))
    if case["case"] == "cabinet_1":
        mapping=load(ROOT / "artifacts/terminal_mapping_topology_20261001/cabinet_1_draft.json")
        diagnostic=[]
        for port in mapping["ports"]:
            x1,y1,x2,y2=port["bbox_xyxy"]
            distances=[]
            for item in shifted["records"]:
                if not item["geometry_pair_eligible"]:
                    continue
                for point in item["visible_ends_xy"]:
                    dx=max(x1-point[0],0,point[0]-x2);dy=max(y1-point[1],0,point[1]-y2)
                    distances.append({"record_id":item["record_id"],"point":point,"distance_to_old_roi_px":float(np.hypot(dx,dy))})
            diagnostic.append({"port_draft_id":port["id"],"bbox_unchanged":port["bbox_xyxy"],
                "nearest_diagnostics_not_assignments":sorted(distances,key=lambda x:(x["distance_to_old_roi_px"],x["record_id"]))[:3],
                "confirmed":False,"automatic_connections_emitted":False})
        (OUT / "old_roi_distance_audit.json").write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT / "group_summary.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
print(json.dumps(reports,indent=2))
