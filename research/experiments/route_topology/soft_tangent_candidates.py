"""Tolerant local geometry, separate inferred candidates and observed pixels."""
import math
from copy import deepcopy
import numpy as np
from tangent_gap_candidates import describe_fragments

PARAMS=dict(lengths=[6,9,13],max_reverse_deg=100.,max_width_ratio=4.,max_gap_widths=24.,
            handles=[.2,1/3,.5],max_turn_deg=165.,max_score=.50,margin=.12)


def direction_evidence(endpoint):
    output=deepcopy(endpoint);xy=np.asarray(endpoint['support_points_xy'],float);vectors=[]
    for n in PARAMS['lengths']:
        if len(xy)<n:continue
        v=xy[0]-xy[:n][n//2:].mean(axis=0);length=np.linalg.norm(v)
        if length>1e-9:vectors.append(v/length)
    if not vectors:raise ValueError('qualified local support required')
    mean=np.asarray(vectors).mean(axis=0);concentration=float(np.linalg.norm(mean))
    if concentration<1e-9:raise ValueError('cancelled tangent direction')
    # Fewer scales means less observed support, not greater confidence.
    reliability=float(max(.25,min(1.,concentration*len(vectors)/3)))
    output.update(outward_unit=(mean/concentration).tolist(),direction_reliability=reliability,
                  direction_scales=len(vectors),scale_directions=np.asarray(vectors).tolist())
    return output


def geometry(a,b):
    if a['component']==b['component']:return None
    p=np.asarray(a['point_xy'],float);q=np.asarray(b['point_xy'],float)
    u=np.asarray(a['outward_unit'],float);v=np.asarray(b['outward_unit'],float)
    gap=float(np.linalg.norm(q-p));wa=a['width'];wb=b['width']
    if gap<=1e-9 or min(wa,wb)<=0 or not np.isfinite([gap,wa,wb]).all():return None
    ratio=max(wa,wb)/min(wa,wb);scaled=gap/((wa+wb)/2)
    if ratio>PARAMS['max_width_ratio'] or scaled>PARAMS['max_gap_widths']:return None
    d=(q-p)/gap
    angles=[math.degrees(math.acos(float(np.clip(np.dot(u,d),-1,1)))),
            math.degrees(math.acos(float(np.clip(np.dot(v,-d),-1,1))))]
    if max(angles)>PARAMS['max_reverse_deg']:return None
    reliability=min(a['direction_reliability'],b['direction_reliability'])
    direction_weight=.45*reliability;total_weight=direction_weight+.25+.15+.15
    costs=dict(direction=min(1.,sum(angles)/180),distance=min(1.,scaled/24),width=min(1.,math.log(ratio)/math.log(4)))
    variants=[];t=np.linspace(0,1,33)[:,None]
    for handle in PARAMS['handles']:
        c1=p+u*gap*handle;c2=q+v*gap*handle
        curve=(1-t)**3*p+3*(1-t)**2*t*c1+3*(1-t)*t*t*c2+t**3*q
        steps=np.diff(curve,axis=0);lengths=np.linalg.norm(steps,axis=1)
        if (lengths<1e-9).any() or (steps@d<=0).any():continue
        directions=steps/lengths[:,None]
        turn=float(np.degrees(np.arccos(np.clip((directions[:-1]*directions[1:]).sum(axis=1),-1,1))).sum())
        if turn>PARAMS['max_turn_deg']:continue
        score=float((direction_weight*costs['direction']+.25*costs['distance']+.15*costs['width']+.15*min(1.,turn/150))/total_weight+.1*(1-reliability))
        variants.append(dict(score=score,handle=handle,accumulated_turn_deg=turn,virtual_curve_xy=curve.tolist()))
    if not variants:return None
    chosen=min(variants,key=lambda r:(r['score'],r['handle']))
    return dict(endpoint_ids=[a['endpoint_id'],b['endpoint_id']],components=[a['component'],b['component']],
         gap_pixels=gap,gap_widths=scaled,width_ratio=ratio,angles_deg=angles,
         direction_reliability=reliability,soft_costs=costs,curve_variants=len(variants),**chosen,
         observed_gap_pixels=0,physical_identity_confirmed=False,score_is_probability=False)


def match_fragments(ends):
    edges=[];rankings={e['endpoint_id']:[] for e in ends}
    for i,a in enumerate(ends):
        for b in ends[i+1:]:
            edge=geometry(a,b)
            if edge is None:continue
            index=len(edges);edges.append(edge)
            for endpoint in edge['endpoint_ids']:rankings[endpoint].append((edge['score'],index))
    choices={}
    for endpoint,rows in rankings.items():
        rows.sort()
        choices[endpoint]=rows[0][1] if rows and rows[0][0]<=PARAMS['max_score'] and (len(rows)==1 or rows[1][0]-rows[0][0]>=PARAMS['margin']) else None
    for i,edge in enumerate(edges):
        edge['state']='fragment_pair_candidate' if all(choices[e]==i for e in edge['endpoint_ids']) else 'ambiguous_nonreciprocal_or_low_score'
    return edges


def observe_family(raw,regions):
    labels,old_ends,components,skeleton=describe_fragments(raw)
    ends=[direction_evidence(e) for e in old_ends];edges=match_fragments(ends)
    adj={c['component']:set() for c in components}
    for edge in edges:
        if edge['state']=='fragment_pair_candidate':
            a,b=edge['components'];adj[a].add(b);adj[b].add(a)
    paths=[];seen=set()
    for component in adj:
        if component in seen:continue
        todo=[component];group=set()
        while todo:
            c=todo.pop()
            if c in group:continue
            group.add(c);todo.extend(adj[c]-group)
        seen|=group;selection=np.isin(labels,list(group))
        hits={name:int((selection&r).sum()) for name,r in regions.items()}
        if all(v>=8 for v in hits.values()):paths.append(dict(components=sorted(group),anchor_hits=hits,
                   contains_inferred_gap=len(group)>1,physical_identity_confirmed=False))
    return dict(endpoints=ends,edges=edges,anchor_path_candidates=paths,native_components=components,
        observed_pixels_added=0,electrical_continuity='not_assessed',physical_identity_confirmed=False),skeleton
