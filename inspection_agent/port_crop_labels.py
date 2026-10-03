"""Training-only clipping: retain every visible port label, never label fragments as background."""
import hashlib
from inspection_agent.port_tiling import tile_windows


def labeled_tiles(boxes,width,height):
    result=[]
    for tile_id,(x,y,r,b) in enumerate(tile_windows(width,height)):
        labels=[]
        for index,port in enumerate(boxes):
            if port['source_class'] not in (3,4):continue
            l,t,rr,bb=port['box_xyxy'];left,top=max(l,x),max(t,y);right,bottom=min(rr,r),min(bb,b)
            if left>=right or top>=bottom:continue
            labels.append({'source_box_index':index,'class_id':port['source_class']-3,
                           'box_xyxy_local':[left-x,top-y,right-x,bottom-y],
                           'complete':x<=l and y<=t and rr<=r and bb<=b})
        result.append({'tile_id':tile_id,'window':[x,y,r,b],'labels':labels})
    return result


def training_tiles(image_name,tiles,seed='port-crop-plan-20260929'):
    """All positive tiles + at most one negative tile per source training image.

    This is training sampling only. Never use labels to choose inference/val tiles.
    """
    positive=[t for t in tiles if t['labels']];negative=[t for t in tiles if not t['labels']]
    selected=sorted(negative,key=lambda t:(hashlib.sha256(
        (seed+'|'+image_name+'|'+str(t['tile_id'])).encode()).hexdigest(),t['tile_id']))[:1]
    return sorted(positive+selected,key=lambda t:t['tile_id'])
