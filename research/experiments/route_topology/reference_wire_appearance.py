"""Reference-derived hue/width diagnostics, never physical wire identification."""
import cv2
import numpy as np

POLICY=dict(hue_bins=18,saturation_min=64,value_min=32,palette_mass=.9,
            hue_bin_neighbor_tolerance=1,min_colored_pixels=16,color_support_min=.5,
            endpoint_box_expansion=2,width_multiplier=2,not_wire_core_count=True)


def hsv_features(rgb, selection):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or selection.shape!=rgb.shape[:2]:
        raise ValueError('RGB uint8 and equal-shape selection required')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    colored=selection & (hsv[:,:,1]>=POLICY['saturation_min']) & (hsv[:,:,2]>=POLICY['value_min'])
    bins=(hsv[:,:,0].astype(int)*18//180)
    hist=np.bincount(bins[colored],minlength=18).astype(float)
    return hist,int(colored.sum()),colored,bins


def fit_reference(rgb,selection):
    palette=set();samples=[]
    for gain in [.8,1.,1.2]:
        image=np.clip(rgb.astype(float)*gain,0,255).astype('uint8')
        hist,count,_,_=hsv_features(image,selection)
        if count<16:raise ValueError('reference has too little colored wire evidence')
        order=np.argsort(-hist,kind='stable');mass=0
        for b in order:
            if hist[b]<=0:break
            palette.update([(int(b)-1)%18,int(b),(int(b)+1)%18])
            mass+=hist[b]
            if mass>=.9*count:break
        samples.append(dict(gain=gain,colored_pixels=count,histogram=hist.tolist()))
    distance=cv2.distanceTransform(selection.astype('uint8'),cv2.DIST_L2,5)
    widths=2*distance[selection]
    if len(widths)==0:raise ValueError('empty reference selection')
    return dict(allowed_hue_bins=sorted(palette),reference_samples=samples,
                width_p95=float(np.quantile(widths,.95)),policy=POLICY.copy())


def assess(rgb,mask,reference,coordinate_scale=1.):
    if not np.isfinite(coordinate_scale) or coordinate_scale<=0:raise ValueError('invalid coordinate scale')
    hist,count,colored,bins=hsv_features(rgb,mask)
    fraction=float(hist[reference['allowed_hue_bins']].sum()/count) if count else 0.
    distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
    widths=2*distance[mask]*coordinate_scale
    median=float(np.median(widths)) if len(widths) else None
    color_ok=bool(count>=16 and fraction>=.5)
    overwide=bool(median is not None and median>reference['width_p95']*2)
    state=('reference_color_supported' if color_ok and not overwide else
           'insufficient_colored_evidence' if count<16 else 'appearance_mismatch_or_overwide')
    return dict(state=state,colored_pixels=count,matched_color_fraction=fraction,
        colored_fraction=float(count/mask.sum()) if mask.any() else 0.,
        median_width_reference_pixels=median,overwide=overwide,
        appearance_only_not_physical_identity=True),colored & np.isin(bins,reference['allowed_hue_bins'])
