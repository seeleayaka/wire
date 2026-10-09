"""Native RGB connected color paths; no morphological repair or identity claim."""
import cv2
import numpy as np


def hue_families(allowed):
    allowed=set(allowed)
    if not allowed or any(type(b) is not int or not 0<=b<18 for b in allowed):raise ValueError('18-bin hue palette required')
    groups=[]
    while allowed:
        todo=[min(allowed)];group=[]
        while todo:
            b=todo.pop()
            if b not in allowed:continue
            allowed.remove(b);group.append(b);todo.extend([(b-1)%18,(b+1)%18])
        groups.append(sorted(group))
    return sorted(groups)


def observe_paths(rgb,regions,families):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('RGB uint8 required')
    if len(regions)!=2 or any(m.shape!=rgb.shape[:2] or m.dtype!=bool for m in regions.values()):raise ValueError('two bound boolean endpoint regions required')
    flat=[b for group in families for b in group]
    if len(flat)!=len(set(flat)):raise ValueError('overlapping hue families cannot double-count')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)*18//180
    colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32);results=[];overlay=np.zeros(rgb.shape[:2],bool)
    for family in families:
        raw=colored&np.isin(bins,family)
        n,labels,stats,_=cv2.connectedComponentsWithStats(raw.astype('uint8'),connectivity=8)
        witnesses=[]
        for component in range(1,n):
            mask=labels==component
            hits={k:int((mask&region).sum()) for k,region in regions.items()}
            if all(v>=8 for v in hits.values()):
                witnesses.append(dict(component_index=component,pixels=int(stats[component,cv2.CC_STAT_AREA]),anchor_hits=hits))
                overlay|=mask
        results.append(dict(hue_bins=family,qualifying_components=witnesses,native_component_count=n-1))
    count=sum(bool(r['qualifying_components']) for r in results)
    return dict(state='native_multicolor_path_candidate' if count>=2 else 'insufficient_native_color_paths',
        qualifying_hue_families=count,families=results,physical_identity_confirmed=False,
        electrical_continuity='not_assessed',pixels_repaired=False,independent_model_observer_count=1),overlay
