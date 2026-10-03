"""Primary five unchanged, optional five additional high-score multi-tile cues."""
import copy
from core_port_precision_policy import iou,select


def complete(raw, shape):
    height,width=shape;l,t,r,b=raw['box_xyxy']
    return l>16 and t>16 and r<width-16 and b<height-16


def supported_tiles(candidate, raw):
    return sorted({p['source_tile'] for p in raw['edge_kept_predictions']
        if p['confidence']>.5 and p['class_id']==candidate['class_id']
        and iou(p['box_xyxy'],candidate['box_xyxy'])>=.5})


def supplement(raw):
    primary=select(raw,'high_score_complete_frame')
    rows=sorted(raw['merged_predictions'],key=lambda p:(-p['confidence'],
        (p['box_xyxy'][2]-p['box_xyxy'][0])*(p['box_xyxy'][3]-p['box_xyxy'][1]),*p['box_xyxy'],p['class_id']))
    additional=[]
    for row in rows:
        if row['confidence']<=.75 or not complete(row,raw['source_shape']):continue
        if any(iou(row['box_xyxy'],p['box_xyxy'])>=.5 for p in primary+additional):continue
        votes=supported_tiles(row,raw)
        if len(votes)<2:continue
        selected=copy.deepcopy(row);selected['strong_support_tiles']=votes
        additional.append(selected)
        if len(additional)==5:break
    return dict(primary=copy.deepcopy(primary),supplementary=additional,
        all_predictions=copy.deepcopy(primary+additional))
