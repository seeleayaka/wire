"""Fixed three-image annotation import audit and input-contract tests."""
import json
from pathlib import Path
import unittest
from PIL import Image, ImageDraw
from terminal_annotation_inventory import import_inventory, parse_annotations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/terminal_map_audit_20261001"
HEADER = "width;height;class;xmin;ymax;xmax;ymin\n"


class ContractTests(unittest.TestCase):
    def test_y_column_order_and_body_not_port(self):
        result = parse_annotations(HEADER + "10;20;UT 4;5;25;15;5", (100, 100))
        self.assertEqual(result["objects"][0]["bbox_xyxy"], [5, 5, 15, 25])
        self.assertEqual(result["objects"][0]["port_rois"], [])
        self.assertIsNone(result["expected_connections"])
        self.assertEqual(result["topology_assessment"]["status"], "insufficient_evidence")

    def test_accessory_bridge_unknown_are_not_nodes(self):
        for label, kind in [("End Cover", "accessory"), ("CLIPFIX 35", "accessory"),
                            ("Plug-in Bridge", "bridge_object"), ("unseen type", "unknown_object")]:
            with self.subTest(label=label):
                result = parse_annotations(HEADER + f"10;20;{label};5;25;15;5", (100, 100))
                self.assertEqual(result["objects"][0]["kind"], kind)
                self.assertEqual(result["confirmed_connections"], [])

    def test_bad_inputs_rejected(self):
        for data in ["wrong header", HEADER, HEADER + "10;20;UT 4;-1;25;9;5",
                     HEADER + "10;20;UT 4;5;125;15;105", HEADER + "9;20;UT 4;5;25;15;5",
                     HEADER + "10;20;UT 4;15;25;5;5", HEADER + "nan;20;UT 4;5;25;15;5",
                     HEADER + "10;20;UT 4;5;25;15", HEADER + "10;20;UT 4;5;25;15;5;extra"]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_annotations(data, (100, 100))

    def test_ids_stable_under_row_reordering(self):
        rows = ["10;20;UT 4;30;25;40;5", "10;20;UT 4;5;25;15;5"]
        results = [parse_annotations(HEADER + "\n".join(order), (100, 100)) for order in [rows, rows[::-1]]]
        self.assertEqual([(x["id"], x["bbox_xyxy"]) for x in results[0]["objects"]],
                         [(x["id"], x["bbox_xyxy"]) for x in results[1]["objects"]])


def run_audit():
    records = []
    colors = {"terminal_body": "#39df68", "bridge_object": "#ff5656", "accessory": "#ffdc55", "unknown_object": "#bbbbbb"}
    for stem in ["terminal_strip_001_1", "terminal_strip_002_1", "terminal_strip_003_1"]:
        image_path = OUT / (stem + ".jpg")
        inventory = import_inventory(image_path, OUT / (stem + ".csv"))
        (OUT / (stem + "_inventory.json")).write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
        with Image.open(image_path) as image:
            overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for obj in inventory["objects"]:
            box = obj["bbox_xyxy"]
            draw.rectangle(box, outline=colors[obj["kind"]], width=2)
            # Stagger small labels to keep tightly packed terminal identities legible.
            if obj["kind"] == "terminal_body":
                number = int(obj["id"].rsplit("_", 1)[1])
                y = 260 + (number % 4) * 22
                draw.line((box[0] + 3, y + 12, box[0] + 3, box[1]), fill=colors[obj["kind"]])
                draw.text((box[0], y), f"T{number:02}", fill=colors[obj["kind"]])
        draw.rectangle((0, 0, overlay.width, 80), fill="#101824")
        draw.text((20, 15), "SUPPLIED OBJECT LABELS - NOT DETECTOR OUTPUT / NOT PORT ROIS", fill="white")
        draw.text((20, 40), "Green: terminal bodies | Yellow: accessories | Red: bridge objects | Topology: INSUFFICIENT EVIDENCE", fill="white")
        overlay.save(OUT / (stem + "_overlay.png"))
        records.append({"image": stem, "counts": inventory["counts"], "status": inventory["topology_assessment"]["status"], "source": inventory["source"]})
    (OUT / "audit.json").write_text(json.dumps({"selection": "first view of first three named samples; fixed before import", "samples": records}, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    run_audit()
