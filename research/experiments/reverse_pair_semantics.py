"""TRAIN augmentation: normal reference observed against faulty expected crop.

Only positive TRAIN examples get reversed OTHER twins, in their original fold.
This transformation has no source IDs, coordinates, labels or thresholds at
inference. Actual labels and original training rows are never changed.
"""
import torch


def reverse_features(features):
    if features.ndim != 2 or features.shape[1] != 6144:
        raise ValueError('Expected original 6144D paired vectors')
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite paired vectors')
    observed, expected, absolute, product = features.split(1536, dim=1)
    return torch.cat((expected, observed, absolute, product), dim=1)


def augment(features, labels, folds):
    reverse_features(features)  # Validate even a corpus without positives.
    if labels.ndim != 1 or folds.ndim != 1 or len(labels) != len(features) or len(folds) != len(features):
        raise ValueError('Training row correspondence mismatch')
    if labels.dtype != torch.long or folds.dtype != torch.long:
        raise ValueError('Integer labels and folds required')
    if not ((labels >= 0) & (labels <= 2)).all() or not ((folds >= 0) & (folds <= 2)).all():
        raise ValueError('Invalid labels or source folds')
    indices = torch.where(labels > 0)[0]
    return (torch.cat((features, reverse_features(features[indices]))),
            torch.cat((labels, torch.zeros(len(indices), dtype=labels.dtype))),
            torch.cat((folds, folds[indices])), indices)
