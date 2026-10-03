"""Optional, label-free supervised port hints inside frozen review parents."""
from __future__ import annotations

import copy
import math

KEYS = ('left', 'top', 'right', 'bottom')


def _box(box):
    try:
        values = tuple(float(box[k]) for k in KEYS)
    except (KeyError, TypeError, ValueError): return None
    if not all(math.isfinite(v) for v in values): return None
    l,t,r,b = values
    return values if 0 <= l < r and 0 <= t < b else None


def _area(values): return (values[2]-values[0])*(values[3]-values[1])


def select_port_state_hints(parents, predictions, threshold):
    """Preserve parents; never use source classes/labels to select spatial hints."""
    if not isinstance(threshold, (float,int)) or isinstance(threshold,bool) or not math.isfinite(threshold) or not .25 <= threshold <= 1:
        raise ValueError('threshold must be finite and within .25..1')
    parent_boxes = [_box(p) for p in parents]
    if any(b is None for b in parent_boxes): raise ValueError('invalid parent geometry')
    audit = {'prediction_count': len(predictions), 'invalid':0,'below_threshold':0,
             'invalid_warp':0,'no_containing_smaller_parent':0,'eligible':0}
    by_parent = {}; seen = set()
    for prediction in predictions:
        bounds = _box(prediction)
        score, coverage, cls = (prediction.get(k) for k in ('confidence','valid_warp_fraction','class_id'))
        if bounds is None or type(cls) is not int or cls not in (0,1) or any(
            isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1
            for v in (score,coverage)):
            audit['invalid']+=1; continue
        if score <= threshold: audit['below_threshold']+=1; continue
        if coverage < .98: audit['invalid_warp']+=1; continue
        containing = [( _area(p), index) for index,p in enumerate(parent_boxes)
                      if p[0]<=bounds[0] and p[1]<=bounds[1] and p[2]>=bounds[2] and p[3]>=bounds[3]
                      and _area(bounds)<=.5*_area(p)]
        if not containing: audit['no_containing_smaller_parent']+=1; continue
        _,index = min(containing)
        identity = (*bounds, cls)
        if identity not in seen:
            seen.add(identity); audit['eligible']+=1
        rank = (-score, _area(bounds), *bounds, cls)
        current = by_parent.get(index)
        if current is None or rank<current[0]: by_parent[index]=(rank,prediction)
    hints = []
    for index, (_, prediction) in sorted(by_parent.items()):
        hints.append({'parent_index':index,'box':copy.deepcopy(prediction),
                      'role':'trained_port_state_manual_review_hint',
                      'automatic_fault_verdict':False,
                      'warning':'A local learned cue, not verified seating or electrical continuity.'})
    audit['hint_count']=len(hints)
    return {'parents':copy.deepcopy(parents),'hints':hints,'selection_audit':audit}
