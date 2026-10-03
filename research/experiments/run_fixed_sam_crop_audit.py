"""Frozen upper-right-quadrant two-view diagnostic, not a field benchmark."""
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image

sys.path.insert(0, "E:/PythonProject10")
from inspection_agent.terminal_mapping import image_binding
from sam_crop_geometry import upper_right_half_box

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/sam_crop_coverage_20261001"
OUT.mkdir(parents=True, exist_ok=False)
cases = []
for number in ["1", "2"]:
    source = Path("C:/Users/HUAWEI/Desktop/案例/线材柜子") / number / (number + ".png")
    binding = image_binding(source)
    box = upper_right_half_box(binding["image_size"])
    crop_path = OUT / f"cabinet_{number}_upper_right.png"
    with Image.open(source) as image:
        image.crop(tuple(box)).save(crop_path)
    cases.append({"case": f"cabinet_{number}", "source_binding": binding, "crop_box_xyxy": box,
        "crop_binding": image_binding(crop_path), "fresh_run_directory": str(OUT / f"cabinet_{number}_crop_run")})
protocol = {"schema_version": 1, "status": "running", "selection": "original cabinet views 1 and 2; upper-right quadrant fixed before inference",
    "prompt": "cable", "mask_threshold": .5, "min_component_pixels": 50, "crop_edge_margin_px": 2,
    "no_tile_stitching": True, "independent_cabinets_claimed": False, "cases": cases}
(OUT / "frozen_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
for index, case in enumerate(cases):
    print(f"Starting fixed crop {index+1}/2", flush=True)
    subprocess.run([sys.executable, str(ROOT / "experiments/run_bound_sam_endpoints.py"),
        "--image", case["crop_binding"]["image_path"], "--output", case["fresh_run_directory"]], check=True)
    if image_binding(Path(case["source_binding"]["image_path"])) != case["source_binding"]:
        raise ValueError("original source changed")
    protocol["completed_case_count"] = index + 1
    (OUT / "frozen_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
protocol["status"] = "complete"
(OUT / "frozen_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
print("Fixed crop inference complete", flush=True)
