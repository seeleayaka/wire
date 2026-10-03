"""Bounded high-score additions, separate from production budget-one policy."""
import copy
def bounded_expand(predictions,base_threshold=.25,extra_threshold=.5,maximum=5):
    if not .25<=base_threshold<extra_threshold<=1 or type(maximum) is not int or not 1<=maximum<=5:
        raise ValueError('invalid bounded policy')
    eligible=[p for p in predictions if p['confidence']>base_threshold]
    eligible.sort(key=lambda p:(-p['confidence'],(p['box_xyxy'][2]-p['box_xyxy'][0])*(p['box_xyxy'][3]-p['box_xyxy'][1]),*p['box_xyxy'],p['class_id']))
    if not eligible:return []
    return copy.deepcopy(eligible[:1]+[p for p in eligible[1:] if p['confidence']>extra_threshold][:maximum-1])
