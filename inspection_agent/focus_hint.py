"""Validation-only within-parent hint; never replaces a review candidate."""
from __future__ import annotations
import copy
import numpy as np


def box_area(box):
    return max(0,box['right']-box['left'])*max(0,box['bottom']-box['top'])


def intersection_over_union(a,b):
    intersection=max(0,min(a['right'],b['right'])-max(a['left'],b['left']))*max(0,min(a['bottom'],b['bottom'])-max(a['top'],b['top']))
    denominator=box_area(a)+box_area(b)-intersection
    return intersection/denominator if denominator>0 else 0


def select_focus_hint(parent,members,score,width,height,threshold,*,expected_grid=(21,28)):
    """One non-edge, smaller, corroborated and above-calibration raw member, or none."""
    from tools.probe_local_normal_bank import region_score
    score=np.asarray(score)
    if len(expected_grid)!=2 or min(expected_grid)<=0 or score.shape!=tuple(expected_grid) or not np.isfinite(score).all() or not np.isfinite(threshold) or threshold<0:
        raise ValueError('invalid frozen CNN map/threshold')
    if width<=0 or height<=0:raise ValueError('invalid ROI')
    def valid(box):
        values=[box[k] for k in ('left','top','right','bottom')]
        return all(np.isfinite(values)) and 0<=values[0]<values[2]<=width and 0<=values[1]<values[3]<=height
    if not valid(parent):raise ValueError('invalid parent geometry')
    choices=[]
    for member in members:
        if not valid(member):raise ValueError('invalid raw member geometry')
        if not (parent['left']<=member['left'] and parent['top']<=member['top'] and
                member['right']<=parent['right'] and member['bottom']<=parent['bottom']):
            raise ValueError('member outside parent')
        if member.get('source') not in ('tile','refinement') or member.get('touches_tile_edge'):continue
        ratio=box_area(member)/box_area(parent)
        if ratio>.5:continue
        novelty=float(region_score(member,score,width,height))
        if novelty<=threshold:continue
        support={tile for other in members if set(other['source_tiles']).isdisjoint(member['source_tiles'])
                 and intersection_over_union(member,other)>=.25 for tile in other['source_tiles']}
        if not support:continue
        choices.append((len(support),novelty,member,ratio,sorted(support)))
    if not choices:return None
    support,novelty,member,ratio,tiles=max(choices,key=lambda item:item[:2])
    return {'box':copy.deepcopy(member),'role':'within_parent_manual_review_hint',
            'automatic_fault_verdict':False,'cnn_p95':novelty,'support_count':support,
            'support_tiles':tiles,'parent_area_fraction':ratio,
            'warning':'May highlight normal structures; not a final fault box.'}
