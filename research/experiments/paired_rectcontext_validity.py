"""Future rectangle inference must pass BOTH actual rectangular and square masks."""
from paired_port_semantics import valid_boxes as square_valid_boxes
from paired_rectcontext_features import crop_signature


def rectangular_fraction(box, mask, scale):
    l, t, r, b = crop_signature(box, scale); h, w = mask.shape
    part = mask[max(0, t):min(h, b), max(0, l):min(w, r)] if r > 0 and b > 0 and l < w and t < h else mask[:0, :0]
    return float(part.sum()) / max(1, (r - l) * (b - t))


def joint_valid_boxes(boxes, mask):
    square = square_valid_boxes(boxes, mask)
    return [i for i in square if all(rectangular_fraction(boxes[i], mask, scale) >= .85 for scale in (1.5, 3.))]
