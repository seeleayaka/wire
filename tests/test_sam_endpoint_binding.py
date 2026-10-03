from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

try:
    from inspection_agent.sam_endpoint_binding import bind_current_run_endpoints, file_sha256
except ModuleNotFoundError:
    from sam_endpoint_binding import bind_current_run_endpoints, file_sha256


class EndpointBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.snapshot = self.directory / "source.png"
        Image.new("RGB", (100, 60), "white").save(self.snapshot)
        Image.new("RGB", (100, 60), "white").save(self.directory / "input.jpg")
        Image.new("L", (100, 60), 0).save(self.directory / "mask_001.png")
        self.report = {"input": str(self.snapshot.resolve()), "instance_count": 1,
                       "image_state_cache_reused": False, "scores": [.8]}
        (self.directory / "report.json").write_text(json.dumps(self.report), encoding="utf-8")
        self.manifest = {"schema_version": 1, "status": "inference_inputs_and_outputs_verified", "run_id": "constructed_test",
            "image_binding": {"image_path": str(self.snapshot.resolve()), "image_sha256": file_sha256(self.snapshot),
                              "image_size": [100, 60], "coordinate_frame": "source_image_pixels"},
            "verified_files": {str(p.resolve()): file_sha256(p) for p in [self.directory / "input.jpg", self.directory / "report.json", self.directory / "mask_001.png"]}}
        self.endpoints = {"source_dir": str(self.directory), "geometry_contract_version": 1,
                          "records": [{"source_mask_id": 1, "source_score": .8}]}

    def bind(self, report=None, manifest=None, endpoints=None):
        return bind_current_run_endpoints(endpoints or self.endpoints, report or self.report,
                                          self.snapshot, manifest or self.manifest)

    def test_current_run_gets_binding_but_not_coverage(self):
        result = self.bind()
        self.assertEqual(result["image_binding"], self.manifest["image_binding"])
        self.assertFalse(result["coverage_review"]["complete"])
        self.assertNotIn("image_binding", self.endpoints)

    def test_no_manifest_rejects_legacy_cache(self):
        with self.assertRaises(ValueError):
            bind_current_run_endpoints(self.endpoints, self.report, self.snapshot, {})

    def test_reused_cache_rejected(self):
        report = dict(self.report, image_state_cache_reused=True)
        with self.assertRaises(ValueError):
            self.bind(report=report)

    def test_changed_snapshot_or_mask_rejected(self):
        Image.new("RGB", (100, 60), "black").save(self.snapshot)
        with self.assertRaises(ValueError):
            self.bind()
        Image.new("RGB", (100, 60), "white").save(self.snapshot)
        Image.new("L", (100, 60), 255).save(self.directory / "mask_001.png")
        with self.assertRaises(ValueError):
            self.bind()

    def test_wrong_input_or_parsed_report_rejected(self):
        for mutation in [{"input": str(self.directory / "other.png")}, {"scores": [.9]}]:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.bind(report=dict(self.report, **mutation))

    def test_mask_shape_rejected_even_if_fingerprint_updated(self):
        path = self.directory / "mask_001.png"
        Image.new("L", (50, 30), 0).save(path)
        self.manifest["verified_files"][str(path.resolve())] = file_sha256(path)
        with self.assertRaises(ValueError):
            self.bind()

    def test_record_mask_and_score_must_agree(self):
        for record in [{"source_mask_id": 2, "source_score": .8}, {"source_mask_id": 1, "source_score": .7}]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.bind(endpoints=dict(self.endpoints, records=[record]))

    def test_coordinate_frame_and_geometry_required(self):
        manifest = deepcopy(self.manifest)
        manifest["image_binding"]["coordinate_frame"] = "aligned_reference_pixels"
        with self.assertRaises(ValueError):
            self.bind(manifest=manifest)
        with self.assertRaises(ValueError):
            self.bind(endpoints=dict(self.endpoints, geometry_contract_version=0))


if __name__ == "__main__":
    unittest.main()
