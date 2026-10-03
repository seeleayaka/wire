"""Explicit missing body/ring evidence; never a new cue from absent pixels."""
import numpy as np
from paired_core_ring import descriptor as strict_descriptor,normalize_pair

DIMENSIONS=47
MISSING_REASONS=('Insufficient valid body/ring coverage','Insufficient gradient support','Empty component quadrant')


def descriptor(observed,expected,valid,box):
    try:
        value=strict_descriptor(observed,expected,valid,box)
    except ValueError as error:
        if str(error) not in MISSING_REASONS:raise
        return np.zeros(DIMENSIONS,dtype=np.float32)
    return np.concatenate((value,np.ones(1,np.float32))).astype(np.float32)


def fold_standardization(train,evaluation):
    import torch
    if train.ndim!=2 or train.shape[1]!=DIMENSIONS or evaluation.ndim!=2 or evaluation.shape[1]!=DIMENSIONS or len(train)==0:
        raise ValueError('Invalid descriptor matrix')
    if not torch.isfinite(train).all() or not torch.isfinite(evaluation).all():raise ValueError('Nonfinite descriptor matrix')
    mean=train.mean(0);std=train.std(0,unbiased=False).clamp_min(.01)
    return ((train-mean)/std).clamp(-5,5),((evaluation-mean)/std).clamp(-5,5),mean,std


def force_missing_abstention(probabilities,raw_descriptors):
    import torch
    if probabilities.shape!=(len(raw_descriptors),3) or raw_descriptors.shape!=(len(probabilities),DIMENSIONS):
        raise ValueError('Invalid score/descriptor correspondence')
    if not torch.isfinite(probabilities).all() or not torch.isfinite(raw_descriptors).all():raise ValueError('Nonfinite scores or descriptors')
    present=raw_descriptors[:,-1]
    if not ((present==0)|(present==1)).all():raise ValueError('Invalid evidence-present flag')
    if ((probabilities<0)|(probabilities>1)).any() or not torch.allclose(probabilities.sum(1),torch.ones(len(probabilities)),atol=1e-5):
        raise ValueError('Invalid class probabilities')
    missing=present==0
    if (raw_descriptors[missing]!=0).any():raise ValueError('Nonzero missing-evidence descriptor')
    result=probabilities.clone();result[missing]=torch.tensor([1.,0.,0.],dtype=result.dtype)
    return result
