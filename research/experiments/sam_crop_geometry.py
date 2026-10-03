"""Translate independent cropped segments, refusing crop-truncated pairs."""
from copy import deepcopy
import math


def upper_right_half_box(size):
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(x) is not int or x < 2 for x in size):
        raise ValueError("source dimensions must be integers >=2")
    return [size[0] // 2, 0, size[0], size[1] // 2]


def translate_crop_record(record, crop_box, source_size, component_bounds, edge_margin=2):
    if type(edge_margin) is not int or edge_margin < 0:
        raise ValueError("edge margin must be a nonnegative integer")
    if len(crop_box) != 4 or any(type(x) is not int for x in crop_box):
        raise ValueError("crop must have integer pixel coordinates")
    x1, y1, x2, y2 = crop_box
    if not (0 <= x1 < x2 <= source_size[0] and 0 <= y1 < y2 <= source_size[1]):
        raise ValueError("crop lies outside source")
    width, height = x2 - x1, y2 - y1
    if len(component_bounds) != 4:
        raise ValueError("component bounds must contain four coordinates")
    bx1, by1, bx2, by2 = component_bounds
    if not (0 <= bx1 < bx2 <= width and 0 <= by1 < by2 <= height):
        raise ValueError("component outside crop")
    translated = deepcopy(record)
    for field in ["candidate_tips_xy", "visible_ends_xy"]:
        points = record.get(field)
        if not isinstance(points, list):
            raise ValueError("geometry points missing")
        shifted = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
                raise ValueError("invalid point")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError("point outside crop coordinate frame")
            shifted.append([point[0] + x1, point[1] + y1])
        translated[field] = shifted
    # Test the whole component, not only its selected tips; never stitch tiles.
    touches_edge = (bx1 <= edge_margin or by1 <= edge_margin
                    or bx2 >= width - edge_margin or by2 >= height - edge_margin)
    translated["crop_evidence"] = {"box_xyxy": list(crop_box), "component_bbox_in_crop_xyxy": list(component_bounds),
        "edge_margin_px": edge_margin, "boundary_truncated": touches_edge, "translation_only": True}
    if touches_edge:
        translated["pre_crop_guard_geometry"] = {key: translated[key] for key in
            ["geometry_pair_eligible", "endpoint_status", "visible_ends_xy"]}
        translated["geometry_pair_eligible"] = False
        translated["endpoint_status"] = "crop_boundary_truncated_visible_segment"
        translated["visible_ends_xy"] = []
    return translated
