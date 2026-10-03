from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

from dino_feature_diff_v2 import candidate_components  # noqa: E402
from tiled_dino_review import (  # noqa: E402
    _display_candidates_for_large_roi,
    _group_repetitive_candidates,
    _limit_candidates_for_large_roi,
    _merge_candidates,
    refinement_plan,
    tile_plan,
)


class TiledDinoReviewTests(unittest.TestCase):
    def test_thin_high_score_core_can_be_preserved_for_cross_scale_review(self) -> None:
        score = np.zeros((256, 410), dtype=np.float32)
        score[70:132, 220:263] = 100.0
        score[76:126, 229:232] = 255.0
        score[82:122, 238:241] = 255.0

        ordinary = candidate_components(score)
        preserved = candidate_components(score, preserve_thin_cores=True)

        self.assertEqual(ordinary, [])
        self.assertTrue(any(item.get("candidate_kind") == "thin_core" for item in preserved))

    def test_tile_plan_covers_all_edges_and_obeys_limit(self) -> None:
        (tile_width, tile_height), windows = tile_plan(1017, 665, reference_size=(1080, 712))
        self.assertGreaterEqual(tile_width, 224)
        self.assertGreaterEqual(tile_height, 224)
        self.assertLessEqual(len(windows), 12)
        self.assertEqual(min(window[0] for window in windows), 0)
        self.assertEqual(min(window[1] for window in windows), 0)
        self.assertEqual(max(window[2] for window in windows), 1017)
        self.assertEqual(max(window[3] for window in windows), 665)

    def test_refinement_plan_adds_a_bounded_wider_scale(self) -> None:
        (tile_width, tile_height), windows = refinement_plan(1017, 665, reference_size=(1080, 712))
        self.assertGreater(tile_width, 337)
        self.assertLess(tile_height, 317)
        self.assertLessEqual(len(windows), 9)
        self.assertEqual(max(window[2] for window in windows), 1017)
        self.assertEqual(max(window[3] for window in windows), 665)

    def test_overlapping_tile_candidates_merge_with_provenance(self) -> None:
        candidates = [
            {
                "left": 20, "top": 30, "right": 60, "bottom": 90,
                "area": 1200.0, "difference_score": 180.0,
                "source": "tile", "source_tiles": ["tile_01"], "touches_tile_edge": True,
                "evidence_scores": {"traditional_p90": 80.0, "dino_p90": 0.12, "agreement_p90": 0.55},
            },
            {
                "left": 42, "top": 34, "right": 82, "bottom": 92,
                "area": 1100.0, "difference_score": 210.0,
                "source": "tile", "source_tiles": ["tile_02"], "touches_tile_edge": True,
                "evidence_scores": {"traditional_p90": 120.0, "dino_p90": 0.08, "agreement_p90": 0.70},
            },
        ]
        merged = _merge_candidates(candidates)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source_tiles"], ["tile_01", "tile_02"])
        self.assertEqual(merged[0]["difference_score"], 210.0)
        self.assertEqual(merged[0]["evidence_summary"]["source_tile_count"], 2)
        self.assertEqual(merged[0]["evidence_scores"]["traditional_p90_max"], 120.0)
        self.assertEqual(merged[0]["evidence_scores"]["dino_p90_max"], 0.12)
        self.assertEqual(merged[0]["evidence_scores"]["observation_count"], 2)

    def test_contained_candidates_merge_into_one_final_box(self) -> None:
        outer = {
            "left": 20, "top": 30, "right": 120, "bottom": 150,
            "area": 9000.0, "difference_score": 180.0,
            "source": "tile", "source_tiles": ["tile_01"], "touches_tile_edge": False,
        }
        inner = {
            "left": 48, "top": 52, "right": 76, "bottom": 90,
            "area": 900.0, "difference_score": 205.0,
            "source": "tile", "source_tiles": ["tile_02"], "touches_tile_edge": False,
        }
        merged = _merge_candidates([outer, inner])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source_tiles"], ["tile_01", "tile_02"])

    def test_axis_aligned_touching_candidates_merge_into_one_finding(self) -> None:
        upper = {
            "left": 100, "top": 40, "right": 180, "bottom": 70,
            "area": 1800.0, "difference_score": 170.0,
            "source": "whole_roi", "source_tiles": ["whole_roi"], "touches_tile_edge": False,
        }
        lower = {
            "left": 103, "top": 70, "right": 177, "bottom": 130,
            "area": 3600.0, "difference_score": 210.0,
            "source": "tile", "source_tiles": ["tile_03"], "touches_tile_edge": False,
        }

        merged = _merge_candidates([upper, lower])

        self.assertEqual(len(merged), 1)
        self.assertEqual([merged[0][key] for key in ("left", "top", "right", "bottom")], [100, 40, 180, 130])

    def test_large_roi_only_displays_whole_or_strongly_corroborated_tiles(self) -> None:
        candidate = {
            "left": 1, "top": 1, "right": 10, "bottom": 10, "area": 81.0, "difference_score": 200.0,
            "source_tiles": ["tile_01"],
            "evidence_summary": {"source_tile_count": 1, "whole_roi_overlap": False, "tile_edge_hits": 0, "tile_edge_only": False, "merged_observation_count": 1},
        }
        corroborated = {
            **candidate,
            "source_tiles": ["tile_01", "tile_02", "tile_03", "tile_04"],
            "evidence_summary": {**candidate["evidence_summary"], "source_tile_count": 4},
        }
        displayed, suppressed = _display_candidates_for_large_roi([candidate, corroborated])
        self.assertEqual(displayed, [corroborated])
        self.assertEqual(suppressed, [candidate])

    def test_large_roi_keeps_a_whole_roi_candidate_at_the_lower_right_corner(self) -> None:
        corner_candidate = {
            "left": 900, "top": 620, "right": 1017, "bottom": 665,
            "area": 5000.0, "difference_score": 190.0,
            "source_tiles": ["whole_roi", "tile_12"],
            "evidence_summary": {
                "source_tile_count": 1,
                "whole_roi_overlap": True,
                "tile_edge_hits": 1,
                "tile_edge_only": True,
                "merged_observation_count": 2,
                "touches_roi_edge": True,
                "roi_edges": ["right", "bottom"],
            },
        }
        displayed, suppressed = _display_candidates_for_large_roi([corner_candidate])
        self.assertEqual(displayed, [corner_candidate])
        self.assertEqual(suppressed, [])

    def test_large_roi_promotes_a_candidate_repeated_across_two_scales(self) -> None:
        candidate = {
            "left": 100, "top": 100, "right": 140, "bottom": 160,
            "area": 1800.0, "difference_score": 180.0,
            "source_tiles": ["tile_02", "refine_03"],
            "evidence_summary": {
                "source_tile_count": 2,
                "primary_tile_count": 1,
                "refinement_tile_count": 1,
                "evidence_scale_count": 2,
                "whole_roi_overlap": False,
                "tile_edge_hits": 0,
                "tile_edge_only": False,
                "merged_observation_count": 2,
            },
            "evidence_scores": {"cross_evidence_pixel_ratio_max": 0.60},
        }

        displayed, suppressed = _display_candidates_for_large_roi([candidate])

        self.assertEqual(displayed, [candidate])
        self.assertEqual(suppressed, [])

    def test_large_roi_suppresses_weak_cross_scale_residuals(self) -> None:
        candidate = {
            "left": 100, "top": 100, "right": 140, "bottom": 160,
            "area": 1800.0, "difference_score": 180.0,
            "source_tiles": ["tile_02", "refine_03"],
            "evidence_summary": {
                "source_tile_count": 2,
                "primary_tile_count": 1,
                "refinement_tile_count": 1,
                "evidence_scale_count": 2,
                "whole_roi_overlap": False,
                "tile_edge_hits": 0,
                "tile_edge_only": False,
                "merged_observation_count": 2,
            },
            "evidence_scores": {"cross_evidence_pixel_ratio_max": 0.40},
        }

        displayed, suppressed = _display_candidates_for_large_roi([candidate])

        self.assertEqual(displayed, [])
        self.assertEqual(suppressed, [candidate])

    def test_large_roi_requires_cross_scale_support_for_a_thin_core_only_candidate(self) -> None:
        thin_core = {
            "left": 100, "top": 100, "right": 140, "bottom": 160,
            "area": 1800.0, "difference_score": 180.0,
            "source_tiles": ["tile_01", "tile_02", "tile_03", "tile_04"],
            "evidence_summary": {
                "source_tile_count": 4,
                "primary_tile_count": 4,
                "refinement_tile_count": 0,
                "evidence_scale_count": 1,
                "whole_roi_overlap": False,
                "tile_edge_hits": 0,
                "tile_edge_only": False,
                "merged_observation_count": 4,
                "thin_core_observation_count": 4,
            },
        }
        cross_scale = {
            **thin_core,
            "source_tiles": ["tile_01", "refine_01"],
            "evidence_summary": {
                **thin_core["evidence_summary"],
                "source_tile_count": 2,
                "primary_tile_count": 1,
                "refinement_tile_count": 1,
                "evidence_scale_count": 2,
                "merged_observation_count": 2,
                "thin_core_observation_count": 2,
            },
            "evidence_scores": {"cross_evidence_pixel_ratio_max": 0.60},
        }

        displayed, suppressed = _display_candidates_for_large_roi([thin_core, cross_scale])

        self.assertEqual(displayed, [cross_scale])
        self.assertEqual(suppressed, [thin_core])

    def test_large_roi_candidate_budget_keeps_three(self) -> None:
        candidates = [
            {
                "left": index * 20, "top": 0, "right": index * 20 + 10, "bottom": 10,
                "area": 100.0, "difference_score": float(100 + index),
                "source_tiles": ["whole_roi", f"tile_{index:02d}"],
                "evidence_summary": {
                    "source_tile_count": index,
                    "evidence_scale_count": 2,
                    "whole_roi_overlap": True,
                },
                "evidence_scores": {"cross_evidence_pixel_ratio_max": index / 10.0},
            }
            for index in range(7)
        ]

        displayed, suppressed = _limit_candidates_for_large_roi(candidates)

        self.assertEqual(len(displayed), 3)
        self.assertEqual(len(suppressed), 4)
        self.assertEqual(displayed[0]["evidence_summary"]["source_tile_count"], 6)

    def test_large_roi_budget_collapses_to_one_for_dominant_multiscale_evidence(self) -> None:
        dominant = {
            "left": 200, "top": 100, "right": 300, "bottom": 250,
            "area": 15000.0, "difference_score": 210.0,
            "source_tiles": ["whole_roi", "tile_01", "tile_02", "tile_03", "tile_04", "refine_01", "refine_02", "refine_03"],
            "evidence_summary": {
                "source_tile_count": 8,
                "primary_tile_count": 4,
                "refinement_tile_count": 3,
                "evidence_scale_count": 3,
                "merged_observation_count": 12,
                "thin_core_observation_count": 0,
                "whole_roi_overlap": True,
            },
            "evidence_scores": {"cross_evidence_pixel_ratio_max": 0.95},
        }
        runner_up = {
            **dominant,
            "left": 20,
            "right": 80,
            "source_tiles": ["whole_roi", "tile_01", "refine_01"],
            "evidence_summary": {**dominant["evidence_summary"], "source_tile_count": 2, "primary_tile_count": 1, "refinement_tile_count": 1},
            "evidence_scores": {"cross_evidence_pixel_ratio_max": 0.61},
        }

        displayed, suppressed = _limit_candidates_for_large_roi([dominant, runner_up])

        self.assertEqual(displayed, [dominant])
        self.assertEqual(suppressed, [runner_up])

    def test_large_roi_budget_prioritizes_a_cross_evidence_thin_core(self) -> None:
        ordinary = [
            {
                "left": index * 20, "top": 0, "right": index * 20 + 10, "bottom": 10,
                "area": 1000.0, "difference_score": 200.0,
                "source_tiles": [f"tile_{index:02d}", f"refine_{index:02d}"],
                "evidence_summary": {
                    "source_tile_count": 6,
                    "evidence_scale_count": 2,
                    "merged_observation_count": 6,
                    "thin_core_observation_count": 0,
                    "whole_roi_overlap": False,
                },
                "evidence_scores": {"cross_evidence_pixel_ratio_max": 0.80},
            }
            for index in range(5)
        ]
        thin_core = {
            "left": 200, "top": 200, "right": 230, "bottom": 260,
            "area": 1200.0, "difference_score": 190.0,
            "source_tiles": ["tile_09", "refine_09"],
            "evidence_summary": {
                "source_tile_count": 2,
                "evidence_scale_count": 2,
                "merged_observation_count": 2,
                "thin_core_observation_count": 2,
                "whole_roi_overlap": False,
            },
            "evidence_scores": {"cross_evidence_pixel_ratio_max": 0.60},
        }

        displayed, suppressed = _limit_candidates_for_large_roi([*ordinary, thin_core])

        self.assertIn(thin_core, displayed)
        self.assertEqual(len(displayed), 3)
        self.assertEqual(len(suppressed), 3)

    def test_large_roi_keeps_a_whole_roi_candidate_touching_only_the_left_edge(self) -> None:
        left_edge_candidate = {
            "left": 0, "top": 36, "right": 183, "bottom": 157,
            "area": 14800.0, "difference_score": 225.0,
            "source_tiles": ["whole_roi", "tile_01"],
            "evidence_summary": {
                "source_tile_count": 1,
                "whole_roi_overlap": True,
                "tile_edge_hits": 0,
                "tile_edge_only": False,
                "merged_observation_count": 2,
                "touches_roi_edge": True,
                "roi_edges": ["left"],
            },
        }
        displayed, suppressed = _display_candidates_for_large_roi([left_edge_candidate])
        self.assertEqual(displayed, [left_edge_candidate])
        self.assertEqual(suppressed, [])

    def test_repetitive_vertical_evidence_requires_multiple_tiles_and_becomes_one_group(self) -> None:
        candidates = [
            {
                "left": 20, "top": 30, "right": 52, "bottom": 160,
                "area": 3000.0, "difference_score": 175.0, "raw_fine_density": 0.24,
                "source": "repetitive_tile", "source_tiles": ["tile_01"], "touches_tile_edge": False,
            },
            {
                "left": 22, "top": 145, "right": 54, "bottom": 280,
                "area": 3200.0, "difference_score": 182.0, "raw_fine_density": 0.27,
                "source": "repetitive_tile", "source_tiles": ["tile_02"], "touches_tile_edge": True,
            },
            {
                "left": 310, "top": 20, "right": 345, "bottom": 115,
                "area": 2100.0, "difference_score": 165.0, "raw_fine_density": 0.25,
                "source": "repetitive_tile", "source_tiles": ["tile_03"], "touches_tile_edge": False,
            },
        ]
        grouped = _group_repetitive_candidates(candidates, 400, 320)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["source_tiles"], ["tile_01", "tile_02"])
        self.assertEqual(grouped[0]["evidence_summary"]["evidence_kind"], "repetitive_structure")
        self.assertEqual(grouped[0]["evidence_summary"]["repeat_orientation"], "vertical")


if __name__ == "__main__":
    unittest.main()
