"""Normal-fit-only stable dark socket core; appearance novelty remains unconfirmed.

No hand-picked test-image pixels. A center-bound normal-derived region is an
appearance weighting mask, NOT a cable mask or proof of an occupied connector.
"""
import cv2
import numpy as np

POLICY={'source_fit_only':True,'median_gray_dark_quantile':.25,
    'gray_mad_stable_quantile':.5,'core_contains_scope_center':True,
    'min_active_pixels':50,'derivative_guard_radius':1,'grid':[8,4],
    'statistical_outlier_is_not_connector_absence':True}


def learn_core(patches):
    patches=np.asarray(patches)
    if patches.ndim!=4 or patches.shape[1:]!=(50,100,3) or patches.dtype!=np.uint8 or len(patches)<20:
        raise ValueError('at least20 actual registered normal RGB socket patches required')
    gray=np.stack([cv2.cvtColor(p,cv2.COLOR_RGB2GRAY) for p in patches]).astype(np.float64)/255
    median=np.median(gray,axis=0);mad=np.median(np.abs(gray-median),axis=0)
    dark=float(np.quantile(median,.25));stable=float(np.quantile(mad,.5))
    # Strict darkness excludes a large tied background at the quantile. Using
    # <= can otherwise select nearly the whole patch instead of a dark core.
    eligible=((median<dark)&(mad<=stable)).astype(np.uint8)
    count,labels=cv2.connectedComponents(eligible,connectivity=8)
    center_label=int(labels[25,50])
    if center_label==0:raise ValueError('source-derived stable dark core does not contain calibrated scope center')
    selected=(labels==center_label).astype(np.uint8)
    # Only select appearance pixels with a fully selected3x3 neighborhood. This
    # ensures wire/color changes OUTSIDE the learned core cannot enter Sobel.
    guarded=cv2.erode(selected,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    if int(guarded.sum())<50:raise ValueError('too little source-derived stable internal appearance support')
    return guarded.astype(bool),{'normal_fit_patches':len(patches),'dark_threshold':dark,'stability_threshold':stable,
        'source_components':count-1,'selected_center_component':center_label,
        'selected_pixels_before_derivative_guard':int(selected.sum()),'active_pixels':int(guarded.sum()),
        'policy':dict(POLICY),'plug_semantics_verified':False}


def descriptor(rgb,mask):
    rgb=np.asarray(rgb);mask=np.asarray(mask)
    if rgb.shape!=(50,100,3) or rgb.dtype!=np.uint8 or mask.shape!=(50,100) or mask.dtype!=bool or mask.sum()<50:
        raise ValueError('actual registered RGB patch and nonempty learned core required')
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float64)/255
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float64)/255
    dx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)/8;dy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)/8
    magnitude=np.minimum(1,np.hypot(dx,dy));angle=np.arctan2(dy,dx)%np.pi
    result=[]
    for y in range(4):
        for x in range(8):
            ys=slice(y*50//4,(y+1)*50//4);xs=slice(x*100//8,(x+1)*100//8);active=mask[ys,xs]
            if not active.any():result.extend([0.]*10);continue
            region=lab[ys,xs][active];result.extend(region.mean(axis=0));result.extend(region.std(axis=0))
            for k in range(4):result.append(float((magnitude[ys,xs][active]*((angle[ys,xs][active]>=k*np.pi/4)&
                (angle[ys,xs][active]<(k+1)*np.pi/4))).mean()))
    return np.asarray(result,np.float64)
