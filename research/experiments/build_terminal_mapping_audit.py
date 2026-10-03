"""Fixed-source, draft-only mapping audit plus constructed geometry integration."""
import importlib.util
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/terminal_mapping_topology_20261001"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "E:/PythonProject10")
spec = importlib.util.spec_from_file_location("inspection_agent.terminal_mapping", Path(__file__).with_name("terminal_mapping.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


records = []
for name in ["001", "002", "003"]:
    parent = ROOT / "artifacts/terminal_map_audit_20261001"
    image = parent / f"terminal_strip_{name}_1.jpg"
    inventory = json.loads((parent / f"terminal_strip_{name}_1_inventory.json").read_text(encoding="utf-8"))
    mapping = module.create_mapping_draft(image, "terminal_strip_photo", inventory)
    assessment = module.review_mapped_topology(mapping, image)
    assert mapping["ports"] == [] and assessment["decision"] == "insufficient_evidence"
    save(f"strip_{name}_draft.json", mapping)
    save(f"strip_{name}_assessment.json", assessment)
    records.append({"case": f"strip_{name}", "object_count": len(mapping["body_objects"]),
                    "port_count": 0, "decision": assessment["decision"], "binding": mapping["image_binding"]})

for number in ["1", "2", "5"]:
    image = Path("C:/Users/HUAWEI/Desktop/案例/线材柜子") / number / (number + ".png")
    mapping = module.create_mapping_draft(image, "cabinet_reference_photo")
    mapping["scope"] = {"description": "Draft visible terminal mapping; cabinet identity and wiring plan unconfirmed",
                        "expected_complete": False}
    if number == "1":
        # Operator-assistance draft, not manufacturer identity or reviewed ports.
        mapping["ports"] = [{"id": f"upper_strip_candidate_{i}_lower_entry", "device_id": "upper_strip_unconfirmed",
            "terminal_label": f"visible_label_{i}_draft", "roi_kind": "wire_entry_port", "bbox_xyxy": box,
            "confirmed": False, "reviewer": None,
            "evidence_note": "Photo-visible numbered strip; approximate lower wire-entry ROI, identity and seating pending review",
            "wire_label": None} for i, box in [(1, [317, 163, 329, 179]), (2, [327, 164, 338, 181]), (3, [337, 165, 348, 182])]]
        with Image.open(image) as opened:
            overlay = opened.convert("RGB")
            crop = overlay.crop((300, 120, 380, 210)).resize((640, 720))
            crop.save(OUT / "cabinet_1_upper_strip_detail.png")
        draw = ImageDraw.Draw(overlay)
        for port in mapping["ports"]:
            draw.rectangle(port["bbox_xyxy"], outline="#ffb020", width=2)
        draw.rectangle((0, 0, overlay.width, 38), fill="#141b25")
        draw.text((8, 8), "3 DRAFT PORT ROIs - NOT CONFIRMED / NO CONNECTION EDGES", fill="white")
        overlay.save(OUT / "cabinet_1_draft_overlay.png")
    assessment = module.review_mapped_topology(mapping, image)
    assert assessment["decision"] == "insufficient_evidence" and assessment["observed_graph"] is None
    save(f"cabinet_{number}_draft.json", mapping)
    save(f"cabinet_{number}_assessment.json", assessment)
    records.append({"case": f"cabinet_{number}", "port_count": len(mapping["ports"]),
                    "decision": assessment["decision"], "binding": mapping["image_binding"]})
save("fixed_source_audit.json", {"selection": "previously viewed first three terminal samples and original cabinet sources 1,2,5",
    "samples": records, "not_accuracy_metrics": True, "no_sam_inference_executed": True})
print(json.dumps(records, ensure_ascii=False, indent=2))
