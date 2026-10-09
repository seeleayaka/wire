"""Supplemental native RGB components seeded by one trusted SAM bundle."""
import cv2
import numpy as np


def supplement(rgb,native,allowed_bins):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10
    colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)&np.isin(bins,allowed_bins)
    n,labels=cv2.connectedComponents(colored.astype('uint8'),connectivity=8)
    selected=np.zeros(native.shape,bool);components=[]
    for i in range(1,n):
        pixels=labels==i;overlap=int((pixels&native).sum())
        # Same area floor as the established fragment extractor, no case-specific seed.
        if overlap<24:continue
        selected|=pixels;components.append(dict(component=i,pixels=int(pixels.sum()),native_seed_pixels=overlap,
            recovered_RGB_pixels=int((pixels&~native).sum()),physical_identity_confirmed=False))
    return selected,dict(components=components,recovered_RGB_pixels=int((selected&~native).sum()),
        original_SAM_pixels_modified=0,RGB_support_not_SAM_confirmation=True,physical_identity_confirmed=False)
