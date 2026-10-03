"""Mask-evidence audit only: no physical cable identity or connection inference."""
import hashlib
import math
import numpy as np

def audit_duplicates(masks, records, overlap_threshold=.8):
    if len(masks)!=len(records):raise ValueError('record_mask_count_mismatch')
    if not 0<overlap_threshold<=1:raise ValueError('invalid_overlap_threshold')
    normalized=[np.asarray(mask,dtype=bool) for mask in masks]
    if normalized and (any(m.ndim!=2 or m.shape!=normalized[0].shape or not m.any() for m in normalized)):
        raise ValueError('invalid_component_mask')
    ids=[r['record_id'] for r in records]
    if len(set(ids))!=len(ids):raise ValueError('duplicate_record_id')
    groups={}
    for i,mask in enumerate(normalized):
        key=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        groups.setdefault(key,[]).append(i)
    exact=[dict(record_ids=[ids[i] for i in indices],mask_pixels=int(normalized[indices[0]].sum()))
           for indices in groups.values() if len(indices)>1]
    pairs=[];contained=[];overlaps=[]
    for i,a in enumerate(normalized):
        for j in range(i+1,len(normalized)):
            b=normalized[j];intersection=int((a&b).sum());union=int((a|b).sum())
            score=intersection/union
            smaller_overlap=intersection/min(int(a.sum()),int(b.sum()))
            if intersection:
                overlaps.append(dict(record_ids=[ids[i],ids[j]],mask_iou=score,smaller_mask_overlap=smaller_overlap,
                                     action='review_only_do_not_merge'))
            if overlap_threshold<=score<1:
                pairs.append(dict(record_ids=[ids[i],ids[j]],mask_iou=score,
                                  action='review_only_do_not_merge'))
            if smaller_overlap>=overlap_threshold and score<overlap_threshold:
                contained.append(overlaps[-1])
    eligible=[i for i,r in enumerate(records) if r.get('geometry_pair_eligible') is True]
    eligible_groups=set(hashlib.sha256(np.packbits(normalized[i]).tobytes()).hexdigest() for i in eligible)
    return dict(component_records=len(records),exact_mask_groups=len(groups),exact_duplicate_groups=exact,
        exact_duplicate_redundancy=sum(len(g['record_ids'])-1 for g in exact),high_overlap_pairs=pairs,
        contained_overlap_pairs=contained,intersecting_pairs=len(overlaps),
        maximum_pair_mask_iou=max((p['mask_iou'] for p in overlaps),default=0),
        eligible_records=len(eligible),eligible_exact_mask_groups=len(eligible_groups),
        claim='Pixel evidence equivalence only, not physical wire count',
        automatic_connections=[],input_records_changed=False)

def endpoint_port_diagnostics(records,ports):
    result=[]
    for port in ports:
        l,t,r,b=port['bbox_xyxy'];distances=[]
        for record in records:
            if record.get('geometry_pair_eligible') is not True:continue
            for point in record.get('visible_ends_xy',[]):
                x,y=point
                distance=math.hypot(max(l-x,0,x-r),max(t-y,0,y-b))
                distances.append(dict(record_id=record['record_id'],point=point,distance_px=distance))
        distances.sort(key=lambda row:(row['distance_px'],row['record_id'],row['point']))
        result.append(dict(port_id=port['id'],identity_confirmed=port.get('confirmed') is True,
            endpoint_contacts=[d for d in distances if d['distance_px']==0],
            closest_geometry_diagnostic=distances[:3],assignment=None,
            warning='Distances are not terminal identity or a connection verdict'))
    return result
