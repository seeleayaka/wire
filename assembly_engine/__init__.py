"""Template-driven visual assembly inspection primitives.

The package deliberately stays independent from the existing PyQt prototype
chain. It models automatic candidates, observations, and explainable rules;
callers may supply aligned images and optional DINO evidence from any UI.
"""

from .assembly_template import AssemblyTemplate, TemplateObject, TemplateRule, load_template, save_template
from .observation import observe_template
from .reference_builder import build_reference_template
from .rule_engine import evaluate_rules
from .assembly_rule_engine import evaluate_rules as evaluate_assembly_rules

__all__ = [
    "AssemblyTemplate",
    "TemplateObject",
    "TemplateRule",
    "build_reference_template",
    "evaluate_rules",
    "evaluate_assembly_rules",
    "load_template",
    "observe_template",
    "save_template",
]
