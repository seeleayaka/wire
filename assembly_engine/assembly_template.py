"""Versioned data model for an automatically built assembly template."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class TemplateValidationError(ValueError):
    """Raised when an assembly-template JSON document is malformed."""


def _number_list(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise TemplateValidationError(f"{name} must contain exactly {size} values")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise TemplateValidationError(f"{name} must contain numbers") from error


def _region(value: Any, name: str) -> list[float]:
    region = _number_list(value, 4, name)
    left, top, right, bottom = region
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise TemplateValidationError(f"{name} must be normalized [left, top, right, bottom]")
    return region


def _segment(value: Any, name: str) -> list[float] | None:
    if value is None:
        return None
    segment = _number_list(value, 4, name)
    if not all(0.0 <= component <= 1.0 for component in segment):
        raise TemplateValidationError(f"{name} must contain normalized coordinates")
    return segment


@dataclass(frozen=True)
class TemplateObject:
    """One automatic visual candidate in reference-image coordinates."""

    id: str
    kind: str
    color: dict[str, Any]
    center_norm: list[float]
    direction_deg: float
    expected_region: list[float]
    presence_threshold: float
    position_tolerance_px: float
    direction_tolerance_deg: float
    confidence: str = "automatic_candidate"
    candidate_confidence: float = 0.0
    reference_metrics: dict[str, float] = field(default_factory=dict)
    reference_line_norm: list[float] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemplateObject":
        if not isinstance(value, dict):
            raise TemplateValidationError("object must be an object")
        object_id = value.get("id")
        kind = value.get("kind")
        if not isinstance(object_id, str) or not object_id:
            raise TemplateValidationError("object.id must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise TemplateValidationError(f"object {object_id}.kind must be a non-empty string")
        color = value.get("color")
        if not isinstance(color, dict):
            raise TemplateValidationError(f"object {object_id}.color must be an object")
        bgr = _number_list(color.get("bgr"), 3, f"object {object_id}.color.bgr")
        tolerance = float(color.get("tolerance", 48.0))
        if not (0.0 < tolerance <= 255.0):
            raise TemplateValidationError(f"object {object_id}.color.tolerance must be within (0, 255]")
        normalized_color = {
            "name": str(color.get("name", "unknown")),
            "bgr": [round(component, 3) for component in bgr],
            "tolerance": tolerance,
        }
        centre = _number_list(value.get("center_norm"), 2, f"object {object_id}.center_norm")
        if not all(0.0 <= component <= 1.0 for component in centre):
            raise TemplateValidationError(f"object {object_id}.center_norm must be normalized")
        region = _region(value.get("expected_region"), f"object {object_id}.expected_region")
        presence = float(value.get("presence_threshold", 0.60))
        candidate_confidence = float(value.get("candidate_confidence", 0.0))
        if not (0.0 <= presence <= 1.0 and 0.0 <= candidate_confidence <= 1.0):
            raise TemplateValidationError(f"object {object_id} confidence values must be within [0, 1]")
        metrics = value.get("reference_metrics", {})
        if not isinstance(metrics, dict):
            raise TemplateValidationError(f"object {object_id}.reference_metrics must be an object")
        return cls(
            id=object_id,
            kind=kind,
            color=normalized_color,
            center_norm=centre,
            direction_deg=float(value.get("direction_deg", 0.0)) % 180.0,
            expected_region=region,
            presence_threshold=presence,
            position_tolerance_px=float(value.get("position_tolerance_px", 15.0)),
            direction_tolerance_deg=float(value.get("direction_tolerance_deg", 18.0)),
            confidence=str(value.get("confidence", "automatic_candidate")),
            candidate_confidence=candidate_confidence,
            reference_metrics={str(key): float(metric) for key, metric in metrics.items()},
            reference_line_norm=_segment(value.get("reference_line_norm"), f"object {object_id}.reference_line_norm"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemplateRule:
    """One explainable rule that evaluates a visual template object."""

    id: str
    type: str
    object_id: str
    severity: str = "suspected_ng"
    enabled: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemplateRule":
        if not isinstance(value, dict):
            raise TemplateValidationError("rule must be an object")
        rule_id, rule_type, object_id = value.get("id"), value.get("type"), value.get("object_id")
        if not all(isinstance(item, str) and item for item in (rule_id, rule_type, object_id)):
            raise TemplateValidationError("rule id, type, and object_id must be non-empty strings")
        if rule_type not in {"exists", "color", "position"}:
            raise TemplateValidationError(f"unsupported rule type: {rule_type}")
        return cls(
            id=rule_id,
            type=rule_type,
            object_id=object_id,
            severity=str(value.get("severity", "suspected_ng")),
            enabled=bool(value.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssemblyTemplate:
    """A single cabinet model built from one approved reference image."""

    template_id: str
    reference_image: str
    image_size: list[int]
    build_mode: str
    objects: list[TemplateObject]
    rules: list[TemplateRule]
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AssemblyTemplate":
        if not isinstance(value, dict):
            raise TemplateValidationError("template must be a JSON object")
        version = int(value.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise TemplateValidationError(f"unsupported schema_version: {version}")
        template_id = value.get("template_id")
        reference_image = value.get("reference_image")
        build_mode = value.get("build_mode")
        if not all(isinstance(item, str) and item for item in (template_id, reference_image, build_mode)):
            raise TemplateValidationError("template_id, reference_image, and build_mode must be non-empty strings")
        image_size = _number_list(value.get("image_size"), 2, "image_size")
        if not all(component > 0 and float(component).is_integer() for component in image_size):
            raise TemplateValidationError("image_size must be positive integer [width, height]")
        raw_objects = value.get("objects", [])
        raw_rules = value.get("rules", [])
        if not isinstance(raw_objects, list) or not isinstance(raw_rules, list):
            raise TemplateValidationError("objects and rules must be arrays")
        objects = [TemplateObject.from_dict(item) for item in raw_objects]
        object_ids = {item.id for item in objects}
        if len(object_ids) != len(objects):
            raise TemplateValidationError("object ids must be unique")
        rules = [TemplateRule.from_dict(item) for item in raw_rules]
        if len({item.id for item in rules}) != len(rules):
            raise TemplateValidationError("rule ids must be unique")
        unknown_objects = sorted({rule.object_id for rule in rules if rule.object_id not in object_ids})
        if unknown_objects:
            raise TemplateValidationError(f"rules refer to unknown objects: {', '.join(unknown_objects)}")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TemplateValidationError("metadata must be an object")
        return cls(
            schema_version=version,
            template_id=template_id,
            reference_image=reference_image,
            image_size=[int(component) for component in image_size],
            build_mode=build_mode,
            objects=objects,
            rules=rules,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "reference_image": self.reference_image,
            "image_size": self.image_size,
            "build_mode": self.build_mode,
            "objects": [item.to_dict() for item in self.objects],
            "rules": [item.to_dict() for item in self.rules],
            "metadata": self.metadata,
        }

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for item in self.objects:
            by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "reference_image": self.reference_image,
            "image_size": self.image_size,
            "object_count": len(self.objects),
            "object_kinds": by_kind,
            "enabled_rule_count": sum(rule.enabled for rule in self.rules),
            "build_mode": self.build_mode,
        }


def load_template(path: str | Path) -> AssemblyTemplate:
    source = Path(path)
    try:
        return AssemblyTemplate.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except OSError as error:
        raise TemplateValidationError(f"cannot read template {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise TemplateValidationError(f"invalid JSON in {source}: {error.msg}") from error


def save_template(template: AssemblyTemplate, path: str | Path) -> None:
    target = Path(path)
    validated = AssemblyTemplate.from_dict(template.to_dict())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(validated.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
