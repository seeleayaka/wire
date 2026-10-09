"""Image->verified-mask->path->port relation->paired output, plus tamper tests.

All fixtures are constructed. Their artificial SAM manifests are marked as
software fixtures, never actual inference evidence or real human annotations.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from core import image_binding, sha256
from run_review import execute, verified_run
from test_core import mask, ports, REVIEW


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def create_fixture(root, name, pixels):
    run = root / name
    sam = run / "sam"
    sam.mkdir(parents=True)
    snapshot = run / "source.png"
    Image.fromarray(pixels).convert("RGB").save(snapshot)
    Image.fromarray(pixels).save(sam / "mask_001.png")
    Image.fromarray(pixels).convert("RGB").save(sam / "input.jpg")
    save(sam / "report.json", {"input": str(snapshot), "instance_count": 1,
                                "scores": [.9], "image_state_cache_reused": False})
    manifest = {"schema_version": 1, "status": "complete",
                "evidence_kind": "constructed_software_fixture_NOT_model_inference",
                "image_binding": image_binding(snapshot), "recipe": {"no_registration": True},
                "verified_files": {str(p.resolve()): sha256(p) for p in sam.iterdir()}}
    save(run / "run_manifest.json", manifest)
    side = {"image_path": str(snapshot), "image_binding": image_binding(snapshot),
            "ports": ports(), "mask_ids": [1], "scope_description": "constructed one-wire test",
            "selection_review": REVIEW, "coverage_review": REVIEW}
    return run, side


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ref_run, ref = create_fixture(self.root, "ref", mask([(20, 30), (100, 30)]))
        self.ins_run, ins = create_fixture(self.root, "ins", mask([(20, 30), (20, 60), (100, 60), (100, 30)]))
        self.pair = {"schema_version": 1, "scene_type": "software_test", "reference": ref,
                     "inspection": ins, "expected_connections": [{"from": "A", "to": "B"}],
                     "expected_review": REVIEW}
        self.pair_path = self.root / "pair.json"
        save(self.pair_path, self.pair)

    def tearDown(self):
        self.temp.cleanup()

    def run_pair(self, output="out"):
        return execute(self.pair_path, self.ref_run, self.ins_run, self.root / output)

    def test_end_to_end_same_endpoints(self):
        initial = sha256(self.pair_path)
        r = self.run_pair()
        self.assertEqual(r["decision"], "same_visible_terminal_relations")
        self.assertEqual(initial, sha256(self.pair_path))
        self.assertTrue((self.root / "out/reference_paths.png").is_file())
        self.assertFalse(r["inspection"]["provenance"]["fresh_inference_this_review"])

    def test_old_output_never_overwritten(self):
        self.run_pair()
        digest = sha256(self.root / "out/report.json")
        with self.assertRaises(FileExistsError):
            self.run_pair()
        self.assertEqual(digest, sha256(self.root / "out/report.json"))

    def test_mask_drift_rejected_before_output(self):
        Image.fromarray(np.zeros((120, 120), np.uint8)).save(self.ins_run / "sam/mask_001.png")
        with self.assertRaises(ValueError):
            self.run_pair()
        self.assertFalse((self.root / "out").exists())

    def test_image_annotation_hash_drift(self):
        self.pair["inspection"]["image_binding"]["image_sha256"] = "0" * 64
        save(self.pair_path, self.pair)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_wrong_image_coordinates(self):
        self.pair["inspection"]["image_binding"]["coordinate_frame"] = "aligned_pixels"
        save(self.pair_path, self.pair)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_unknown_mask_id(self):
        self.pair["inspection"]["mask_ids"] = [2]
        save(self.pair_path, self.pair)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_duplicate_mask_id(self):
        self.pair["inspection"]["mask_ids"] = [1, 1]
        save(self.pair_path, self.pair)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_no_implicit_manual_selection_confirmation(self):
        self.pair["inspection"]["selection_review"] = {"confirmed": False}
        save(self.pair_path, self.pair)
        self.assertEqual(self.run_pair()["decision"], "insufficient_evidence")

    def test_incomplete_origin_run_rejected(self):
        path = self.ins_run / "run_manifest.json"
        m = json.loads(path.read_text())
        m["status"] = "running"
        save(path, m)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_unverified_warp_rejected(self):
        path = self.ins_run / "run_manifest.json"
        m = json.loads(path.read_text())
        m["recipe"]["no_registration"] = False
        save(path, m)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_expected_unknown_endpoint_rejected(self):
        self.pair["expected_connections"][0]["to"] = "unknown"
        save(self.pair_path, self.pair)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_node_identity_must_match_both_views(self):
        self.pair["inspection"]["ports"][0]["id"] = "different"
        save(self.pair_path, self.pair)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_mask_dimensions_rejected_even_when_hash_updated(self):
        path = self.ins_run / "sam/mask_001.png"
        Image.new("L", (80, 80)).save(path)
        manifest_path = self.ins_run / "run_manifest.json"
        m = json.loads(manifest_path.read_text())
        m["verified_files"][str(path.resolve())] = sha256(path)
        save(manifest_path, m)
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_disconnected_instance_not_split_to_claim_connection(self):
        broken = mask([(20, 30), (100, 30)])
        broken[:, 50:70] = 0
        run, side = create_fixture(self.root, "broken", broken)
        self.ins_run, self.pair["inspection"] = run, side
        save(self.pair_path, self.pair)
        self.assertEqual(self.run_pair()["decision"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)
