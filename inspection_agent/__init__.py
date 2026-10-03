"""Auditable inspection-agent primitives shared by cabinet and bench demos."""

from .topology import ConnectionGraph, TopologyValidationError, compare_topologies
from .image2net_adapter import Image2NetAdapterError, connection_graph_from_image2net
from .sam_topology_adapter import SamTopologyAdapterError, connection_graph_from_sam_endpoints
from .normal_reference import (
    NormalReferenceError,
    classification_metrics,
    descriptor_from_patch_features,
    nearest_normal_score,
    select_balanced_threshold,
)
from .workflow import InspectionTask, WorkflowError

__all__ = [
    "ConnectionGraph",
    "InspectionTask",
    "Image2NetAdapterError",
    "SamTopologyAdapterError",
    "NormalReferenceError",
    "TopologyValidationError",
    "WorkflowError",
    "compare_topologies",
    "connection_graph_from_image2net",
    "connection_graph_from_sam_endpoints",
    "classification_metrics",
    "descriptor_from_patch_features",
    "nearest_normal_score",
    "select_balanced_threshold",
]
