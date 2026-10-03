"""Fresh offline SAM -> independent visible endpoints -> image-bound review.

Always uses a new output directory and a byte-identical source snapshot. No
homography, cache migration, graph completion or field accuracy claim.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "inspection_agent").is_dir():
    sys.path.insert(0, str(ROOT))
    from inspection_agent.sam_endpoint_binding import file_sha256, bind_current_run_endpoints
else:
    sys.path.insert(0, "E:/PythonProject10")
    from sam_endpoint_binding import file_sha256, bind_current_run_endpoints
from inspection_agent.terminal_mapping import image_binding, review_mapped_topology


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("E:/PythonProject10"))
    args = parser.parse_args()
    project = args.project_root.resolve()
    source = args.image.resolve()
    if not source.suffix or not source.is_file():
        raise ValueError("source must be a saved image")
    binding = image_binding(source)
    runner = project / "manual_review/run_sam3_cable_probe.py"
    extractor = project / "manual_review/extract_sam3_visible_segment_endpoints.py"
    checkpoint = project / "models/sam3/sam3.pt"
    sam_python = project / "runtime/sam3/.venv/Scripts/python.exe"
    for path in [runner, extractor, checkpoint, sam_python]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / ("source_snapshot" + source.suffix.lower())
    shutil.copyfile(source, snapshot)
    if file_sha256(snapshot) != binding["image_sha256"]:
        raise ValueError("source changed during snapshot")
    runtime = {str(p.resolve()): file_sha256(p) for p in [runner, extractor, checkpoint,
        project / "inspection_agent/visible_segment_geometry.py", project / "inspection_agent/terminal_mapping.py"]}
    manifest = {"schema_version": 1, "run_id": args.output.resolve().name,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running_fresh_inference",
        "image_binding": binding, "runtime_fingerprints": runtime,
        "recipe": {"prompt": "cable", "threshold": .5, "threads": 8, "min_component_pixels": 50,
                   "no_cache_reuse": True, "no_registration": True}}
    save(args.output / "run_manifest.json", manifest)
    sam_dir = args.output / "sam"
    # Runner's default cache lies in this fresh directory and therefore cannot exist.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project / "runtime/sam3/source")
    environment["HF_HUB_OFFLINE"] = "1"
    print("Starting fresh SAM on byte-identical source snapshot", flush=True)
    try:
        with (args.output / "sam_stdout.log").open("w", encoding="utf-8") as stdout, (args.output / "sam_stderr.log").open("w", encoding="utf-8") as stderr:
            subprocess.run([str(sam_python), str(runner), "--input", str(snapshot), "--checkpoint", str(checkpoint),
                "--output-dir", str(sam_dir), "--prompt", "cable", "--threshold", "0.5", "--threads", "8"],
                env=environment, stdout=stdout, stderr=stderr, check=True)
        if image_binding(source) != binding or file_sha256(snapshot) != binding["image_sha256"]:
            raise ValueError("source identity changed while inference was running")
        for path, sha in runtime.items():
            if file_sha256(Path(path)) != sha:
                raise ValueError("runtime changed during inference")
        sam_report = load(sam_dir / "report.json")
        artifact_paths = [sam_dir / "report.json", sam_dir / "input.jpg"] + sorted(sam_dir.glob("mask_*.png"))
        verified = {str(p.resolve()): file_sha256(p) for p in artifact_paths}
        manifest.update(status="inference_inputs_and_outputs_verified", verified_files=verified)
        save(args.output / "run_manifest.json", manifest)
        endpoint_dir = args.output / "endpoints"
        if sam_report.get("instance_count") == 0:
            endpoint_dir.mkdir()
            raw_endpoints = {"source_dir": str(sam_dir.resolve()), "geometry_contract_version": 1,
                "source_prompt": "cable", "source_confidence_threshold": .5, "min_component_pixels": 50, "records": []}
            save(endpoint_dir / "endpoint_report.json", raw_endpoints)
        else:
            # Same extractor and geometry rules, in this process with captured arguments.
            spec = importlib.util.spec_from_file_location("current_run_endpoint_extractor", extractor)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            previous_args = sys.argv
            try:
                sys.argv = [str(extractor), "--source-dir", str(sam_dir), "--output-dir", str(endpoint_dir), "--min-component-pixels", "50"]
                module.main()
            finally:
                sys.argv = previous_args
            raw_endpoints = load(endpoint_dir / "endpoint_report.json")
        bound = bind_current_run_endpoints(raw_endpoints, sam_report, snapshot, manifest)
        save(args.output / "bound_endpoint_report.json", bound)
        if args.map:
            assessment = review_mapped_topology(load(args.map), source, bound)
            save(args.output / "topology_review.json", assessment)
        manifest.update(status="complete", endpoint_record_count=len(bound["records"]),
            geometry_eligible_count=sum(r.get("geometry_pair_eligible") is True for r in bound["records"]),
            automatic_fault_verdict=False, observation_coverage_confirmed=False)
        save(args.output / "run_manifest.json", manifest)
        print(json.dumps({"status": "complete", "instances": sam_report["instance_count"],
            "endpoint_records": len(bound["records"]), "eligible": manifest["geometry_eligible_count"],
            "output": str(args.output.resolve())}), flush=True)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        save(args.output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
