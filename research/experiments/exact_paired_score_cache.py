"""Reuse ONLY byte-bound identical semantic inputs, never another model vote.

Detector role/pose provenance may change; paired crop classifier input does not
depend on it. Changed pixels, reference warp, runtime or exact box always miss.
The caller must verify cached outputs against a completed independent audit.
"""
from copy import deepcopy
import hashlib
import json
import math


def binding(source_sha,reference_sha,runtime,alignment,shape):
    for digest in (source_sha,reference_sha):
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('actual SHA256 identity required')
    if len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape):raise ValueError('actual source frame required')
    if (not isinstance(runtime,dict) or not runtime.get('encoder') or not runtime.get('head')
        or not isinstance(alignment,dict) or alignment.get('alignment_quality',{}).get('reliable') is not True):
        raise ValueError('frozen classifier and reliable warp required')
    matrix=alignment.get('source_to_reference_homography')
    if (not isinstance(matrix,list) or len(matrix)!=3 or any(not isinstance(row,list) or len(row)!=3 for row in matrix)
        or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for row in matrix for v in row)):
        raise ValueError('finite actual reference warp required')
    # All runtime/code and alignment fields are bound, not only the head name.
    value=dict(source_sha256=source_sha,reference_sha256=reference_sha,runtime=runtime,
               alignment=alignment,source_shape=list(shape))
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def key(candidate):
    cls=candidate['class_id'];box=candidate['box_xyxy']
    if type(cls) is not int or cls not in (0,1) or len(box)!=4 or any(
        isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
        raise ValueError('finite class/box required')
    if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('ordered crop required')
    # No IoU, rounding, nearest crop or confidence-selected cache hit.
    return (cls,*box)


def reuse(proposals,old_proposals,old_probabilities,old_binding,new_binding):
    if len(old_proposals)!=len(old_probabilities):raise ValueError('cached proposal/probability count mismatch')
    cached={}
    for candidate,probability in zip(old_proposals,old_probabilities):
        if (len(probability)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) or not 0<=v<=1 for v in probability) or abs(sum(probability)-1)>1e-5):
            raise ValueError('invalid cached probability vector')
        identity=key(candidate)
        if identity in cached and cached[identity]!=probability:raise ValueError('contradictory exact semantic cache')
        cached[identity]=probability
    scores=[];missing=[];reused=0
    for index,candidate in enumerate(proposals):
        identity=key(candidate)
        value=cached.get(identity) if old_binding==new_binding else None
        scores.append(deepcopy(value))
        if value is None:missing.append(index)
        else:reused+=1
    return scores,missing,reused
