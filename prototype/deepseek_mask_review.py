"""Manual DeepSeek review of two binary cable masks.

The local pipeline remains authoritative.  This module never sends cabinet
photos, file paths, OCR, device metadata, or a local decision.  It only lets a
manually invoked external reviewer rank local candidate IDs and give a short
Chinese explanation for an operator.
"""
from __future__ import annotations

import base64
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "deepseek_mask_review.json"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DeepSeekMaskReviewError(RuntimeError):
    """External review failure that must not affect the local result."""


@dataclass(frozen=True)
class DeepSeekMaskReviewSettings:
    enabled: bool
    api_key_env: str
    local_key_file: Path
    endpoint: str
    model: str
    timeout_seconds: float
    max_tokens: int
    max_mask_bytes: int

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "manual_only": True,
            "provider": "deepseek",
            "credential_sources": [self.api_key_env, self.local_key_file.name],
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "payload_policy": "two_binary_mask_union_pngs_and_minimal_local_candidate_evidence_only",
            "not_sent": ["raw_images", "file_paths", "ocr", "device_identifiers", "user_metadata", "local_decision"],
            "not_authoritative_for": ["candidate_boxes", "local_decision", "fault_type", "terminal_assignment", "electrical_continuity"],
        }


def load_settings(config_path: Path = DEFAULT_CONFIG) -> DeepSeekMaskReviewSettings:
    if not config_path.is_file():
        raise DeepSeekMaskReviewError(f"DeepSeek mask-review configuration is missing: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DeepSeekMaskReviewError("DeepSeek mask-review configuration is invalid JSON") from error
    endpoint = str(raw.get("endpoint", "")).strip()
    model = str(raw.get("model", "")).strip()
    api_key_env = str(raw.get("api_key_env", "DEEPSEEK_API_KEY")).strip()
    local_key_file = config_path.parent / str(raw.get("local_key_file", "deepseek_mask_review.local.json")).strip()
    if not endpoint.startswith("https://") or not model or not api_key_env or local_key_file.parent != config_path.parent:
        raise DeepSeekMaskReviewError("DeepSeek mask-review configuration has no valid HTTPS endpoint or model")
    return DeepSeekMaskReviewSettings(
        enabled=bool(raw.get("enabled", False)),
        api_key_env=api_key_env,
        local_key_file=local_key_file,
        endpoint=endpoint,
        model=model,
        timeout_seconds=max(1.0, min(float(raw.get("timeout_seconds", 45)), 120.0)),
        max_tokens=max(128, min(int(raw.get("max_tokens", 1800)), 4096)),
        max_mask_bytes=max(1024, min(int(raw.get("max_mask_bytes", 4 * 1024 * 1024)), 16 * 1024 * 1024)),
    )


def _local_api_key(settings: DeepSeekMaskReviewSettings) -> str:
    """Read an ignored project-local key file without ever reporting its value."""
    if not settings.local_key_file.is_file():
        return ""
    try:
        local = json.loads(settings.local_key_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DeepSeekMaskReviewError("project-local DeepSeek key configuration is invalid JSON") from error
    value = local.get("api_key") if isinstance(local, dict) else None
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DeepSeekMaskReviewError("project-local DeepSeek key must be a string")
    return value.strip()


def save_local_api_key(settings: DeepSeekMaskReviewSettings, api_key: str) -> None:
    """Persist a manually entered key in the ignored project-local configuration."""
    token = api_key.strip()
    if not token:
        raise DeepSeekMaskReviewError("DeepSeek API key cannot be empty")
    payload: dict[str, Any] = {}
    if settings.local_key_file.is_file():
        try:
            existing = json.loads(settings.local_key_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DeepSeekMaskReviewError("project-local DeepSeek key configuration is invalid JSON") from error
        if not isinstance(existing, dict):
            raise DeepSeekMaskReviewError("project-local DeepSeek key configuration must be an object")
        payload.update(existing)
    payload["api_key"] = token
    temporary_path = settings.local_key_file.with_name(settings.local_key_file.name + ".tmp")
    try:
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(settings.local_key_file)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise DeepSeekMaskReviewError("could not save the project-local DeepSeek API key") from error


def _png_dimensions(data: bytes, label: str) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise DeepSeekMaskReviewError(f"{label} is not a valid PNG mask")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise DeepSeekMaskReviewError(f"{label} has invalid image dimensions")
    return width, height


def _read_mask(path: Path, label: str, maximum_bytes: int) -> tuple[bytes, tuple[int, int]]:
    if not path.is_file():
        raise DeepSeekMaskReviewError(f"{label} mask is unavailable")
    if path.stat().st_size > maximum_bytes:
        raise DeepSeekMaskReviewError(f"{label} mask exceeds the configured size limit")
    data = path.read_bytes()
    return data, _png_dimensions(data, label)


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DeepSeekMaskReviewError(f"local candidate has invalid {name}")
    return float(value)


def _candidate_xyxy(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    """Accept the legacy DINO rectangle or the SAM3 fusion rectangle contract."""
    bbox_xyxy = candidate.get("bbox_xyxy")
    if bbox_xyxy is not None:
        if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
            raise DeepSeekMaskReviewError("local candidate has invalid bbox_xyxy")
        return tuple(
            _number(value, f"bbox_xyxy[{index}]")
            for index, value in enumerate(bbox_xyxy)
        )  # type: ignore[return-value]
    return (
        _number(candidate.get("left"), "left"),
        _number(candidate.get("top"), "top"),
        _number(candidate.get("right"), "right"),
        _number(candidate.get("bottom"), "bottom"),
    )


def _candidate_payload(candidates: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    if not candidates:
        raise DeepSeekMaskReviewError("there are no local candidates to rank")
    payload: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        left, top, right, bottom = _candidate_xyxy(candidate)
        if not (0.0 <= left < right <= width and 0.0 <= top < bottom <= height):
            raise DeepSeekMaskReviewError("local candidate is outside the SAM3 mask frame")
        item: dict[str, Any] = {
            "candidate_id": f"candidate_{index:03d}",
            "bbox_normalized_xyxy": [round(left / width, 5), round(top / height, 5), round(right / width, 5), round(bottom / height, 5)],
        }
        for key in ("difference_score", "area", "component_pixels"):
            value = candidate.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                item[key] = round(float(value), 4)
        payload.append(item)
    return payload


def _image_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _api_token(settings: DeepSeekMaskReviewSettings, api_key: str | None) -> str:
    token = (api_key if api_key is not None else os.environ.get(settings.api_key_env, "") or _local_api_key(settings)).strip()
    if not token:
        raise DeepSeekMaskReviewError(f"{settings.api_key_env} and the project-local key configuration are both unavailable")
    return token


def _mask_inputs(
    reference_mask_path: Path,
    inspection_mask_path: Path,
    candidates: list[dict[str, Any]],
    settings: DeepSeekMaskReviewSettings,
) -> tuple[bytes, bytes, list[dict[str, Any]], list[str]]:
    reference_data, reference_size = _read_mask(reference_mask_path, "reference", settings.max_mask_bytes)
    inspection_data, inspection_size = _read_mask(inspection_mask_path, "aligned inspection", settings.max_mask_bytes)
    if reference_size != inspection_size:
        raise DeepSeekMaskReviewError("reference and aligned inspection masks have different dimensions")
    candidate_data = _candidate_payload(candidates, *reference_size)
    return reference_data, inspection_data, candidate_data, [item["candidate_id"] for item in candidate_data]


def build_request_payload(
    reference_mask_path: Path,
    inspection_mask_path: Path,
    candidates: list[dict[str, Any]],
    settings: DeepSeekMaskReviewSettings,
) -> tuple[dict[str, Any], list[str]]:
    """Build an image-only external payload without paths or local decisions."""
    reference_data, inspection_data, candidate_data, candidate_ids = _mask_inputs(
        reference_mask_path, inspection_mask_path, candidates, settings
    )
    content = [
        {
            "type": "text",
            "text": (
                "You are ranking pre-existing visible cable-mask difference candidates for a human reviewer. "
                "The first image is the reference binary cable union mask; the second is the aligned inspection binary cable union mask. "
                "Do not infer terminals, electrical continuity, cable identity, or any fault type. Do not create, merge, remove, or move boxes. "
                "Return JSON only with exactly these keys: ranked_candidate_ids (all IDs once, most worthy of human review first) and review_summary_zh (a concise Chinese explanation, <=240 Chinese characters). "
                f"Local candidates: {json.dumps(candidate_data, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
        {"type": "image_url", "image_url": {"url": _image_url(reference_data)}},
        {"type": "image_url", "image_url": {"url": _image_url(inspection_data)}},
    ]
    return {
        "model": settings.model,
        "messages": [
            {
                "role": "system",
                "content": "Return a conservative ranking for human review. The local computer-vision result remains authoritative. JSON only.",
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": settings.max_tokens,
        "response_format": {"type": "json_object"},
    }, candidate_ids


def build_question_payload(
    reference_mask_path: Path,
    inspection_mask_path: Path,
    candidates: list[dict[str, Any]],
    question_zh: str,
    settings: DeepSeekMaskReviewSettings,
) -> tuple[dict[str, Any], list[str]]:
    """Build a plain-language mask-only question without requiring structured output."""
    question = question_zh.strip()
    if not question or len(question) > 500:
        raise DeepSeekMaskReviewError("DeepSeek question must contain 1 to 500 characters")
    reference_data, inspection_data, candidate_data, candidate_ids = _mask_inputs(
        reference_mask_path, inspection_mask_path, candidates, settings
    )
    content = [
        {
            "type": "text",
            "text": (
                "The first image is a reference binary cable union mask and the second is an aligned inspection binary cable union mask. "
                "The local candidate boxes are normalized coordinates only. Answer the operator in concise Chinese based only on these masks and boxes. "
                "Do not infer terminals, electrical continuity, physical cable identity, wrong connection, missing connection, or any confirmed fault type. "
                f"Operator question: {question}\n"
                f"Local candidates: {json.dumps(candidate_data, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
        {"type": "image_url", "image_url": {"url": _image_url(reference_data)}},
        {"type": "image_url", "image_url": {"url": _image_url(inspection_data)}},
    ]
    return {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": "Give a conservative Chinese answer for human review. The local result remains authoritative."},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": settings.max_tokens,
    }, candidate_ids


def _json_content(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        raise DeepSeekMaskReviewError("DeepSeek response has no text content")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        object_start = text.find("{")
        if object_start < 0:
            raise DeepSeekMaskReviewError("DeepSeek model did not return the required JSON object") from error
        try:
            value, _ = json.JSONDecoder().raw_decode(text[object_start:])
        except json.JSONDecodeError as nested_error:
            raise DeepSeekMaskReviewError("DeepSeek model did not return the required JSON object") from nested_error
    if not isinstance(value, dict):
        raise DeepSeekMaskReviewError("DeepSeek response JSON must be an object")
    return value


def _validate_response(response: dict[str, Any], candidate_ids: list[str]) -> tuple[list[str], str]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise DeepSeekMaskReviewError("DeepSeek response has no completion choice") from error
    output = _json_content(content)
    if set(output) != {"ranked_candidate_ids", "review_summary_zh"}:
        raise DeepSeekMaskReviewError("DeepSeek response must contain only the requested ranking and summary")
    ranking = output["ranked_candidate_ids"]
    summary = output["review_summary_zh"]
    if not isinstance(ranking, list) or not all(isinstance(item, str) for item in ranking):
        raise DeepSeekMaskReviewError("DeepSeek response ranking is invalid")
    if len(ranking) != len(candidate_ids) or set(ranking) != set(candidate_ids):
        raise DeepSeekMaskReviewError("DeepSeek response returned unknown, missing, or duplicate candidate IDs")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 480:
        raise DeepSeekMaskReviewError("DeepSeek response summary is invalid")
    return ranking, summary.strip()


def _response_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise DeepSeekMaskReviewError("DeepSeek response has no completion choice") from error
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekMaskReviewError("DeepSeek response has no text answer")
    return content.strip()


def _post_json(request: Request, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: endpoint is config-validated HTTPS
            status = int(getattr(response, "status", 0) or response.getcode())
            content_type = str(response.headers.get("Content-Type", "unknown")).split(";", 1)[0].strip() or "unknown"
            data = response.read()
    except HTTPError as error:
        raise DeepSeekMaskReviewError(f"DeepSeek request failed (HTTP {error.code})") from error
    except URLError as error:
        raise DeepSeekMaskReviewError("DeepSeek request could not reach the configured endpoint") from error
    except TimeoutError as error:
        raise DeepSeekMaskReviewError("DeepSeek request timed out") from error
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeepSeekMaskReviewError(
            f"DeepSeek response is not valid JSON (HTTP {status}; Content-Type {content_type}; {len(data)} bytes)"
        ) from error
    if not isinstance(payload, dict):
        raise DeepSeekMaskReviewError("DeepSeek returned an invalid response object")
    return payload


def review_masks(
    reference_mask_path: Path,
    inspection_mask_path: Path,
    candidates: list[dict[str, Any]],
    settings: DeepSeekMaskReviewSettings | None = None,
    api_key: str | None = None,
    sender: Callable[[Request, float], dict[str, Any]] = _post_json,
) -> dict[str, Any]:
    """Perform one manual external review and return a report-safe result."""
    settings = settings or load_settings()
    if not settings.enabled:
        raise DeepSeekMaskReviewError("DeepSeek mask review is disabled in configuration")
    token = _api_token(settings, api_key)
    payload, candidate_ids = build_request_payload(reference_mask_path, inspection_mask_path, candidates, settings)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}

    def request_ranking() -> tuple[list[str], str]:
        request = Request(settings.endpoint, data=body, headers=headers, method="POST")
        return _validate_response(sender(request, settings.timeout_seconds), candidate_ids)

    format_retry_count = 0
    try:
        ranking, summary = request_ranking()
    except DeepSeekMaskReviewError as error:
        if str(error) != "DeepSeek model did not return the required JSON object":
            raise
        format_retry_count = 1
        ranking, summary = request_ranking()
    return {
        "status": "ok",
        "manual_only": True,
        "external_review_is_non_authoritative": True,
        "settings": settings.report(),
        "candidate_id_mapping": [
            {"candidate_id": candidate_id, "local_region_index": index}
            for index, candidate_id in enumerate(candidate_ids, 1)
        ],
        "ranked_candidate_ids": ranking,
        "review_summary_zh": summary,
        "format_retry_count": format_retry_count,
        "operator_notice": "仅用于人工复核排序与说明；本地候选框和本地结论未被修改。",
    }


def ask_masks(
    reference_mask_path: Path,
    inspection_mask_path: Path,
    candidates: list[dict[str, Any]],
    question_zh: str,
    settings: DeepSeekMaskReviewSettings | None = None,
    api_key: str | None = None,
    sender: Callable[[Request, float], dict[str, Any]] = _post_json,
) -> dict[str, Any]:
    """Ask one operator-written question about two binary masks and local candidates."""
    settings = settings or load_settings()
    if not settings.enabled:
        raise DeepSeekMaskReviewError("DeepSeek mask review is disabled in configuration")
    token = _api_token(settings, api_key)
    payload, candidate_ids = build_question_payload(
        reference_mask_path, inspection_mask_path, candidates, question_zh, settings
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}

    def request_answer(request_payload: dict[str, Any]) -> str:
        request = Request(
            settings.endpoint,
            data=json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        return _response_text(sender(request, settings.timeout_seconds))

    answer_retry_count = 0
    try:
        answer = request_answer(payload)
    except DeepSeekMaskReviewError as error:
        if str(error) != "DeepSeek response has no text answer":
            raise
        answer_retry_count = 1
        retry_payload = {**payload, "max_tokens": max(1800, int(payload["max_tokens"]))}
        answer = request_answer(retry_payload)
    return {
        "status": "ok",
        "manual_only": True,
        "external_review_is_non_authoritative": True,
        "settings": settings.report(),
        "question_zh": question_zh.strip(),
        "answer_zh": answer,
        "answer_retry_count": answer_retry_count,
        "candidate_count": len(candidate_ids),
        "operator_notice": "仅用于人工复核问答；本地候选框和本地结论未被修改。",
    }
