"""Bounded global photometric gain from registered pairs; no image IDs or GT."""
import numpy as np

def compensate(observed,expected,valid_mask):
    if observed.dtype!=np.uint8 or expected.dtype!=np.uint8 or observed.shape!=expected.shape or observed.ndim!=3 or observed.shape[2]!=3:
        raise ValueError('Expected equally sized uint8 three-channel paired images')
    if valid_mask.shape!=observed.shape[:2]:raise ValueError('Photometric mask geometry mismatch')
    if not np.isfinite(valid_mask).all():raise ValueError('Nonfinite photometric mask')
    # Uniform, photo-independent samples; do not select boxes or fit labels.
    a=observed[::32,::32].reshape(-1,3).astype(np.float64)
    b=expected[::32,::32].reshape(-1,3).astype(np.float64)
    valid=valid_mask[::32,::32].reshape(-1)>.99
    gains=[];dispersion=[];counts=[]
    for channel in range(3):
        keep=valid&(a[:,channel]>=16)&(a[:,channel]<=239)&(b[:,channel]>=16)&(b[:,channel]<=239)
        counts.append(int(keep.sum()))
        if counts[-1]<512:return observed.copy(),dict(status='abstained',reason='insufficient_registered_unsaturated_samples',sample_counts=counts)
        ratios=np.log(b[keep,channel])-np.log(a[keep,channel]);center=float(np.median(ratios))
        gains.append(float(np.exp(center)));dispersion.append(float(np.median(np.abs(ratios-center))))
    evidence=dict(channel_gain=gains,log_ratio_MAD=dispersion,sample_counts=counts,
        method='uniform32_registered_pair_log_median',no_GT_or_metadata=True)
    if any(g<.7 or g>1.4 for g in gains):return observed.copy(),dict(status='abstained',reason='gain_outside_fixed_safe_range',**evidence)
    if max(dispersion)>.12:return observed.copy(),dict(status='abstained',reason='nonuniform_or_unreliable_photometric_pair',**evidence)
    if max(abs(g-1.) for g in gains)<=.025:return observed.copy(),dict(status='identity',reason='already_near_matched_exposure',**evidence)
    result=np.rint(observed.astype(np.float32)*np.asarray(gains,dtype=np.float32)).clip(0,255).astype(np.uint8)
    return result,dict(status='compensated',detector_branch_only=True,never_replace_visual_report_pixels=True,**evidence)
