"""Explainable first-version exists, color, and position assembly rules."""
from __future__ import annotations

from typing import Any

from .assembly_template import AssemblyTemplate, TemplateObject, TemplateRule


def _result(rule: TemplateRule, item: TemplateObject, status: str, reason: str, expected: Any, actual: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": rule.id,
        "type": rule.type,
        "object_id": item.id,
        "status": status,
        "severity": rule.severity,
        "reason": reason,
        "expected": expected,
        "actual": actual,
        "evidence": evidence,
    }


def _external_support(observation: dict[str, Any]) -> bool:
    external = observation.get("evidence", {}).get("external", {})
    dino = external.get("dino_difference")
    conventional = external.get("traditional_difference")
    return bool((isinstance(dino, (float, int)) and dino >= 0.18) or (isinstance(conventional, (float, int)) and conventional >= 0.18))


def _evaluate_exists(rule: TemplateRule, item: TemplateObject, observation: dict[str, Any]) -> dict[str, Any]:
    if not observation.get("observable"):
        return _result(rule, item, "manual_review", "object_region_not_observable", item.presence_threshold, None, observation)
    presence = float(observation["presence_score"])
    traditional_difference = float(observation["traditional_difference"])
    structural_similarity = float(observation.get("structural_similarity", 1.0))
    # Thin wire removals can produce a modest mean pixel difference even when
    # the line evidence disappears completely.  Require both signals so a
    # small lighting change alone does not become a missing-installation claim.
    structural_change = traditional_difference >= 0.16 or (
        traditional_difference >= 0.04
        and structural_similarity <= 0.95
        and float(observation.get("line_strength", 0.0)) < 0.15
    )
    multi_evidence = structural_change and (float(observation["line_strength"]) < 0.35 or _external_support(observation))
    if presence >= item.presence_threshold:
        return _result(rule, item, "pass", "presence_evidence_sufficient", item.presence_threshold, presence, observation)
    if multi_evidence:
        return _result(rule, item, "suspected_ng", "suspected_missing_with_multiple_visual_evidence", item.presence_threshold, presence, observation)
    return _result(rule, item, "manual_review", "presence_evidence_insufficient", item.presence_threshold, presence, observation)


def _evaluate_color(rule: TemplateRule, item: TemplateObject, observation: dict[str, Any]) -> dict[str, Any]:
    if not observation.get("observable"):
        return _result(rule, item, "manual_review", "object_region_not_observable", item.color, None, observation)
    presence = float(observation["presence_score"])
    distance = float(observation["color_distance_lab"])
    expected = {"name": item.color["name"], "tolerance": item.color["tolerance"]}
    if presence < item.presence_threshold:
        return _result(rule, item, "manual_review", "cannot_assert_color_when_presence_is_uncertain", expected, distance, observation)
    if distance <= float(item.color["tolerance"]):
        return _result(rule, item, "pass", "color_within_tolerance", expected, distance, observation)
    line_supported_color_change = (
        float(observation.get("line_strength", 0.0)) >= 0.35
        and distance > float(item.color["tolerance"]) * 1.25
    )
    if float(observation["traditional_difference"]) >= 0.12 or _external_support(observation) or line_supported_color_change:
        return _result(rule, item, "suspected_ng", "suspected_wrong_color_with_difference_evidence", expected, distance, observation)
    return _result(rule, item, "manual_review", "color_distance_high_but_not_independently_supported", expected, distance, observation)


def _evaluate_position(rule: TemplateRule, item: TemplateObject, observation: dict[str, Any]) -> dict[str, Any]:
    if not observation.get("observable") or observation.get("position_error_px") is None:
        return _result(rule, item, "manual_review", "line_position_not_observable", item.position_tolerance_px, None, observation)
    presence = float(observation["presence_score"])
    position_error = float(observation["position_error_px"])
    direction_error = float(observation["direction_error_deg"])
    expected = {"position_tolerance_px": item.position_tolerance_px, "direction_tolerance_deg": item.direction_tolerance_deg}
    actual = {"position_error_px": position_error, "direction_error_deg": direction_error}
    if observation.get("reference_self_check") and float(observation.get("traditional_difference", 1.0)) <= 1e-6:
        return _result(rule, item, "pass", "reference_self_check", expected, actual, observation)
    if presence < item.presence_threshold:
        return _result(rule, item, "manual_review", "cannot_assert_position_when_presence_is_uncertain", expected, actual, observation)
    if position_error <= item.position_tolerance_px and direction_error <= item.direction_tolerance_deg:
        return _result(rule, item, "pass", "position_within_tolerance", expected, actual, observation)
    if float(observation["traditional_difference"]) >= 0.12 or _external_support(observation):
        return _result(rule, item, "suspected_ng", "suspected_position_deviation_with_difference_evidence", expected, actual, observation)
    return _result(rule, item, "manual_review", "position_deviation_not_independently_supported", expected, actual, observation)


def evaluate_rules(template: AssemblyTemplate, observation_report: dict[str, Any]) -> dict[str, Any]:
    """Compare a template with observations and return a machine-readable verdict."""
    if observation_report.get("template_id") != template.template_id:
        raise ValueError("observation report belongs to a different template")
    observations = {item.get("object_id"): item for item in observation_report.get("actual_observations", [])}
    objects = {item.id: item for item in template.objects}
    results: list[dict[str, Any]] = []
    evaluators = {"exists": _evaluate_exists, "color": _evaluate_color, "position": _evaluate_position}
    for rule in template.rules:
        if not rule.enabled:
            continue
        item = objects[rule.object_id]
        observation = observations.get(item.id, {"object_id": item.id, "observable": False, "reason": "observation_missing"})
        results.append(evaluators[rule.type](rule, item, observation))
    counts = {status: sum(result["status"] == status for result in results) for status in ("pass", "suspected_ng", "manual_review")}
    if not template.objects:
        decision, headline = "no_conclusion", "模板未生成足够可靠的自动候选"
    elif counts["suspected_ng"]:
        decision, headline = "suspected_ng", "发现有多项视觉证据支持的疑似装配异常"
    elif counts["manual_review"]:
        decision, headline = "manual_review", "视觉证据不足或存在矛盾，需人工复核"
    else:
        decision, headline = "ok", "已启用的装配规则全部通过"
    return {
        "template_id": template.template_id,
        "decision": decision,
        "headline": headline,
        "summary": {"object_count": len(template.objects), "rule_count": len(results), **counts},
        "rule_results": results,
    }
