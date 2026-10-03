"""Candidate-shaped input pixels, frozen CLS/central patch paired descriptors."""
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
SCALES = (1.5, 3.)


def crop_signature(box, scale):
    l, t, r, b = map(float, box)
    if not all(math.isfinite(v) for v in (l, t, r, b, scale)) or r <= l or b <= t or scale <= 0:
        raise ValueError('invalid rectangular context')
    cx, cy = (l + r) / 2, (t + b) / 2
    width, height = max(4., (r - l) * scale), max(4., (b - t) * scale)
    return (math.floor(cx - width / 2), math.floor(cy - height / 2),
            math.ceil(cx + width / 2), math.ceil(cy + height / 2))


def crop_tensor(image, signature):
    left, top, right, bottom = signature
    h, w = image.shape[:2]
    if left >= w or top >= h or right <= 0 or bottom <= 0:
        raise ValueError('rectangular crop outside image')
    crop = cv2.copyMakeBorder(image[max(0, top):min(h, bottom), max(0, left):min(w, right)],
        max(0, -top), max(0, bottom - h), max(0, -left), max(0, right - w),
        cv2.BORDER_CONSTANT, value=(124, 116, 104))
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.
    return (tensor - torch.tensor([.485, .456, .406]).view(3, 1, 1)) / torch.tensor([.229, .224, .225]).view(3, 1, 1)


def embeddings(model, image, boxes, audit=None):
    requested = [crop_signature(box, scale) for box in boxes for scale in SCALES]
    unique = list(dict.fromkeys(requested)); vectors = []
    with torch.inference_mode():
        for start in range(0, len(unique), 8):
            result = model.forward_features(torch.stack([crop_tensor(image, signature) for signature in unique[start:start + 8]]))
            patches = result['x_norm_patchtokens'].reshape(-1, 16, 16, 384)
            cls = F.normalize(result['x_norm_clstoken'], dim=1)
            center = F.normalize(patches[:, 6:10, 6:10].mean(dim=(1, 2)), dim=1)
            vectors.append(torch.cat((cls, center), dim=1))
    if audit is not None:
        audit.update(unique_crops=len(unique), requested_crops=len(requested), rectangular_pixels=True)
    if not boxes: return torch.empty((0, 1536))
    descriptors = torch.cat(vectors); positions = {signature: i for i, signature in enumerate(unique)}
    result = descriptors[[positions[signature] for signature in requested]].reshape(len(boxes), 1536) / 2.
    if not torch.isfinite(result).all(): raise ValueError('nonfinite rectangle descriptors')
    return result
