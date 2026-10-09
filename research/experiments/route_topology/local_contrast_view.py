"""Frozen photometric view for segmentation, not geometric repair or a vote."""
import cv2
import numpy as np

POLICY=dict(kind='LAB_L_CLAHE',clip_limit=2.0,tile_grid=[8,8],
    geometric_resampling=False,mask_morphology=False,independent_observer=False)

def contrast_view(rgb):
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<8:
        raise ValueError('native uint8 RGB frame of at least8x8 required')
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=POLICY['clip_limit'],tileGridSize=tuple(POLICY['tile_grid'])).apply(lab[:,:,0])
    result=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    assert result.shape==rgb.shape and result.dtype==rgb.dtype
    return result
