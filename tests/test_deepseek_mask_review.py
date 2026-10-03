from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import deepseek_mask_review as review  # noqa: E402


def png_header(width: int, height: int) -> bytes:
    return review.PNG_SIGNATURE + b"\x00\x00\x00\rIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")


class DeepSeekMaskReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = review.DeepSeekMaskReviewSettings(
            enabled=True,
            api_key_env="DEEPSEEK_API_KEY",
            local_key_file=ROOT / "config" / "deepseek_mask_review.local.json",
            endpoint="https://api.deepseek.com/chat/completions",
            model="deepseek-v4-flash-vision-exp",
            timeout_seconds=15.0,
            max_tokens=900,
            max_mask_bytes=1024,
        )
        self.candidates = [
            {"left": 10, "top": 20, "right": 30, "bottom": 50, "difference_score": 18.5, "area": 60},
            {"left": 60, "top": 10, "right": 90, "bottom": 40, "difference_score": 12.2},
        ]

    def _masks(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        directory = tempfile.TemporaryDirectory()
        reference = Path(directory.name) / "reference_mask_union.png"
        inspection = Path(directory.name) / "inspection_mask_union.png"
        reference.write_bytes(png_header(100, 80))
        inspection.write_bytes(png_header(100, 80))
        return directory, reference, inspection

    def test_payload_contains_only_masks_and_minimal_normalized_candidates(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)
        payload, candidate_ids = review.build_request_payload(reference, inspection, self.candidates, self.settings)
        content = payload["messages"][1]["content"]
        self.assertEqual(candidate_ids, ["candidate_001", "candidate_002"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual([item["type"] for item in content], ["text", "image_url", "image_url"])
        self.assertNotIn(str(reference), content[0]["text"])
        self.assertNotIn(str(inspection), content[0]["text"])
        self.assertNotIn("local_decision", content[0]["text"])
        self.assertTrue(base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]).startswith(review.PNG_SIGNATURE))
        self.assertIn("[0.1,0.25,0.3,0.625]", content[0]["text"])

    def test_payload_accepts_sam3_fusion_bbox_xyxy_candidates(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)
        fused_candidates = [
            {
                "id": "green_01",
                "tier": "dino_and_sam",
                "bbox_xyxy": [10, 20, 30, 50],
                "sam_support_pixels": 64,
            },
            {
                "id": "yellow_01",
                "tier": "sam_led_human_review",
                "bbox_xyxy": [60, 10, 90, 40],
                "mask_pixels": 120,
            },
        ]

        payload, candidate_ids = review.build_request_payload(reference, inspection, fused_candidates, self.settings)

        self.assertEqual(candidate_ids, ["candidate_001", "candidate_002"])
        text = payload["messages"][1]["content"][0]["text"]
        self.assertIn("[0.1,0.25,0.3,0.625]", text)
        self.assertIn("[0.6,0.125,0.9,0.5]", text)
        self.assertNotIn("green_01", text)
        self.assertNotIn("yellow_01", text)

    def test_question_payload_allows_plain_chinese_answer_without_raw_images(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)

        payload, candidate_ids = review.build_question_payload(
            reference, inspection, self.candidates, "哪些区域应优先人工检查？", self.settings
        )

        self.assertEqual(candidate_ids, ["candidate_001", "candidate_002"])
        self.assertNotIn("response_format", payload)
        content = payload["messages"][1]["content"]
        self.assertEqual([item["type"] for item in content], ["text", "image_url", "image_url"])
        self.assertNotIn(str(reference), content[0]["text"])
        self.assertNotIn(str(inspection), content[0]["text"])

    def test_question_mode_returns_a_plain_chinese_model_answer(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)

        def sender(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            self.assertNotIn("response_format", body)
            return {"choices": [{"message": {"content": "候选一与候选二都建议人工检查。"}}]}

        result = review.ask_masks(
            reference,
            inspection,
            self.candidates,
            "哪些区域应优先人工检查？",
            self.settings,
            api_key="test-key",
            sender=sender,
        )

        self.assertEqual(result["answer_zh"], "候选一与候选二都建议人工检查。")
        self.assertEqual(result["question_zh"], "哪些区域应优先人工检查？")
        self.assertNotIn("test-key", json.dumps(result, ensure_ascii=False))

    def test_question_mode_retries_with_more_tokens_when_only_reasoning_is_returned(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)
        calls = 0
        constrained = review.DeepSeekMaskReviewSettings(
            **{**self.settings.__dict__, "max_tokens": 400}
        )

        def sender(request, timeout):
            nonlocal calls
            calls += 1
            body = json.loads(request.data.decode("utf-8"))
            if calls == 1:
                self.assertEqual(body["max_tokens"], 400)
                return {"choices": [{"finish_reason": "length", "message": {
                    "content": "", "reasoning_content": "internal reasoning", "role": "assistant"
                }}]}
            self.assertGreaterEqual(body["max_tokens"], 1800)
            return {"choices": [{"finish_reason": "stop", "message": {
                "content": "候选一建议优先人工查看。", "reasoning_content": "", "role": "assistant"
            }}]}

        result = review.ask_masks(
            reference, inspection, self.candidates, "哪些区域应优先人工检查？", constrained,
            api_key="test-key", sender=sender,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(result["answer_retry_count"], 1)
        self.assertEqual(result["answer_zh"], "候选一建议优先人工查看。")

    def test_rejects_malformed_sam3_fusion_bbox_xyxy(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)

        with self.assertRaisesRegex(review.DeepSeekMaskReviewError, "invalid bbox_xyxy"):
            review.build_request_payload(
                reference,
                inspection,
                [{"bbox_xyxy": [10, 20, 30]}],
                self.settings,
            )

    def test_manual_review_accepts_only_a_complete_local_candidate_ranking(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)

        def sender(request, timeout):
            self.assertEqual(timeout, 15.0)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            return {
                "choices": [{"message": {"content": json.dumps({
                    "ranked_candidate_ids": ["candidate_002", "candidate_001"],
                    "review_summary_zh": "候选二的掩膜差异更集中，建议优先人工查看。",
                }, ensure_ascii=False)}}]
            }

        result = review.review_masks(reference, inspection, self.candidates, self.settings, api_key="test-key", sender=sender)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ranked_candidate_ids"], ["candidate_002", "candidate_001"])
        self.assertTrue(result["external_review_is_non_authoritative"])
        self.assertNotIn("test-key", json.dumps(result, ensure_ascii=False))

    def test_retries_once_when_the_model_ignores_json_mode(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)
        calls = 0

        def sender(request, timeout):
            nonlocal calls
            calls += 1
            self.assertEqual(json.loads(request.data.decode("utf-8"))["response_format"], {"type": "json_object"})
            if calls == 1:
                return {"choices": [{"message": {"content": "I cannot provide JSON."}}]}
            return {"choices": [{"message": {"content": json.dumps({
                "ranked_candidate_ids": ["candidate_001", "candidate_002"],
                "review_summary_zh": "已按候选差异强度完成排序。",
            }, ensure_ascii=False)}}]}

        result = review.review_masks(reference, inspection, self.candidates, self.settings, api_key="test-key", sender=sender)

        self.assertEqual(calls, 2)
        self.assertEqual(result["format_retry_count"], 1)
        self.assertEqual(result["ranked_candidate_ids"], ["candidate_001", "candidate_002"])

    def test_accepts_a_json_object_wrapped_in_model_explanation(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)

        def sender(request, timeout):
            return {"choices": [{"message": {"content": (
                "以下是排序结果：\n"
                + json.dumps({
                    "ranked_candidate_ids": ["candidate_002", "candidate_001"],
                    "review_summary_zh": "候选二优先人工复核。",
                }, ensure_ascii=False)
                + "\n请人工确认。"
            )}}]}

        result = review.review_masks(reference, inspection, self.candidates, self.settings, api_key="test-key", sender=sender)

        self.assertEqual(result["format_retry_count"], 0)
        self.assertEqual(result["ranked_candidate_ids"], ["candidate_002", "candidate_001"])

    def test_rejects_unknown_candidate_ids_without_local_mutation(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)

        def sender(request, timeout):
            return {"choices": [{"message": {"content": json.dumps({
                "ranked_candidate_ids": ["candidate_001", "candidate_999"],
                "review_summary_zh": "排序。",
            }, ensure_ascii=False)}}]}

        with self.assertRaisesRegex(review.DeepSeekMaskReviewError, "unknown, missing, or duplicate"):
            review.review_masks(reference, inspection, self.candidates, self.settings, api_key="test-key", sender=sender)

    def test_project_local_key_file_is_ignored_by_git(self) -> None:
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("config/deepseek_mask_review.local.json", ignored)

    def test_project_local_key_file_is_used_without_exposing_it_in_the_result(self) -> None:
        directory, reference, inspection = self._masks()
        self.addCleanup(directory.cleanup)
        config = Path(directory.name) / "deepseek_mask_review.json"
        local = Path(directory.name) / "deepseek_mask_review.local.json"
        config.write_text(json.dumps({
            "enabled": True,
            "api_key_env": "UNSET_TEST_DEEPSEEK_KEY",
            "local_key_file": local.name,
            "endpoint": "https://api.deepseek.com/chat/completions",
            "model": "deepseek-v4-flash-vision-exp",
        }), encoding="utf-8")
        review.save_local_api_key(review.load_settings(config), "local-test-key")
        self.assertEqual(json.loads(local.read_text(encoding="utf-8")), {"api_key": "local-test-key"})

        def sender(request, timeout):
            self.assertEqual(request.get_header("Authorization"), "Bearer local-test-key")
            return {"choices": [{"message": {"content": json.dumps({
                "ranked_candidate_ids": ["candidate_001", "candidate_002"],
                "review_summary_zh": "本机配置密钥的离线模拟响应。",
            }, ensure_ascii=False)}}]}

        result = review.review_masks(reference, inspection, self.candidates, review.load_settings(config), sender=sender)
        self.assertNotIn("local-test-key", json.dumps(result, ensure_ascii=False))

    def test_reports_safe_metadata_for_non_json_provider_response(self) -> None:
        class HtmlResponse:
            status = 200
            headers = {"Content-Type": "text/html; charset=utf-8"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self) -> bytes:
                return b"<html>gateway response</html>"

        request = review.Request("https://api.deepseek.com/chat/completions", method="POST")
        with patch.object(review, "urlopen", return_value=HtmlResponse()):
            with self.assertRaisesRegex(
                review.DeepSeekMaskReviewError,
                r"HTTP 200; Content-Type text/html; 29 bytes",
            ):
                review._post_json(request, 1.0)


if __name__ == "__main__":
    unittest.main()
