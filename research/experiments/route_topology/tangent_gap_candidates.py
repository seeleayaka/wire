"""Local break tangent proposals. Virtual edges NEVER become observed pixels."""
import math
import cv2
import numpy as np
from skimage.morphology import skeletonize

PARAMS=dict(min_area=24,steps=12,min_points=6,border=2,max_width_ratio=2.,
            max_gap_widths=12.,max_angle_deg=35.,max_turn_deg=70.,margin=.15)


def neighbors(point, pixels):
    y,x=point;out=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if not (dy or dx) or (y+dy,x+dx) not in pixels:continue
            if dy and dx and ((y+dy,x) in pixels or (y,x+dx) in pixels):continue
            out.append((y+dy,x+dx))
    return sorted(out)


def describe_fragments(raw):
    if raw.dtype!=bool or raw.ndim!=2:raise ValueError('native boolean mask required')
    n,labels,stats,_=cv2.connectedComponentsWithStats(raw.astype('uint8'),8)
    skeleton=skeletonize(raw);pixels=set(map(tuple,np.argwhere(skeleton)))
    graph={p:neighbors(p,pixels) for p in pixels}
    # Padding makes all-foreground masks finite too; widths are descriptive only.
    distance=cv2.distanceTransform(np.pad(raw.astype('uint8'),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    ends=[];h,w=raw.shape
    for start in sorted(pixels):
        if len(graph[start])!=1:continue
        y,x=start;component=int(labels[y,x])
        if stats[component,cv2.CC_STAT_AREA]<PARAMS['min_area']:continue
        if min(x,y,w-1-x,h-1-y)<PARAMS['border']:continue
        points=[start];previous=None;current=start
        for _ in range(PARAMS['steps']):
            if current!=start and len(graph[current])>2:break
            next_points=[p for p in graph[current] if p!=previous]
            if len(next_points)!=1:break
            previous,current=current,next_points[0];points.append(current)
        if len(points)<PARAMS['min_points']:continue
        xy=np.asarray([(p[1],p[0]) for p in points],float)
        outward=xy[0]-xy[len(xy)//2:].mean(axis=0);norm=np.linalg.norm(outward)
        if norm<1e-9:continue
        width=float(2*np.median([distance[p] for p in points[2:]]))
        ends.append(dict(endpoint_id=len(ends),component=component,point_xy=xy[0].tolist(),
                         outward_unit=(outward/norm).tolist(),width=width,
                         support_points_xy=xy.tolist()))
    components=[dict(component=i,pixels=int(stats[i,cv2.CC_STAT_AREA])) for i in range(1,n)]
    return labels,ends,components,skeleton


def geometry(a,b):
    if a['component']==b['component']:return None
    p=np.asarray(a['point_xy'],float);q=np.asarray(b['point_xy'],float)
    u=np.asarray(a['outward_unit'],float);v=np.asarray(b['outward_unit'],float)
    wa=float(a['width']);wb=float(b['width']);gap=float(np.linalg.norm(q-p))
    if min(wa,wb)<=0 or not all(np.isfinite([wa,wb,gap])) or gap<1e-9:return None
    ratio=max(wa,wb)/min(wa,wb);scaled=gap/((wa+wb)/2)
    if ratio>PARAMS['max_width_ratio'] or scaled>PARAMS['max_gap_widths']:return None
    d=(q-p)/gap
    angles=[math.degrees(math.acos(float(np.clip(np.dot(u,d),-1,1)))),
            math.degrees(math.acos(float(np.clip(np.dot(v,-d),-1,1))))]
    if max(angles)>PARAMS['max_angle_deg']:return None
    c1=p+u*gap/3;c2=q+v*gap/3;t=np.linspace(0,1,33)[:,None]
    curve=(1-t)**3*p+3*(1-t)**2*t*c1+3*(1-t)*t*t*c2+t**3*q
    steps=np.diff(curve,axis=0);lengths=np.linalg.norm(steps,axis=1)
    if (lengths<1e-9).any() or (steps@d<=0).any():return None
    directions=steps/lengths[:,None]
    turn=float(np.degrees(np.arccos(np.clip((directions[:-1]*directions[1:]).sum(axis=1),-1,1))).sum())
    if turn>PARAMS['max_turn_deg']:return None
    score=float((max(angles)/PARAMS['max_angle_deg']+scaled/PARAMS['max_gap_widths']+math.log(ratio)/math.log(2))/3)
    return dict(endpoint_ids=[a['endpoint_id'],b['endpoint_id']],components=[a['component'],b['component']],
                gap_pixels=gap,gap_widths=scaled,width_ratio=ratio,angles_deg=angles,
                accumulated_turn_deg=turn,score=score,virtual_curve_xy=curve.tolist(),
                observed_gap_pixels=0,physical_identity_confirmed=False)


def match_fragments(ends):
    candidates=[];ranking={e['endpoint_id']:[] for e in ends}
    for i,a in enumerate(ends):
        for b in ends[i+1:]:
            edge=geometry(a,b)
            if edge is None:continue
            index=len(candidates);candidates.append(edge)
            for endpoint in edge['endpoint_ids']:ranking[endpoint].append((edge['score'],index))
    choices={}
    for endpoint,rows in ranking.items():
        rows.sort()
        choices[endpoint]=(rows[0][1] if rows and (len(rows)==1 or rows[1][0]-rows[0][0]>=PARAMS['margin']) else None)
    for i,edge in enumerate(candidates):
        edge['state']='fragment_pair_candidate' if all(choices[e]==i for e in edge['endpoint_ids']) else 'ambiguous_or_nonreciprocal'
    return candidates


def observe_family(raw,regions):
    labels,ends,components,skeleton=describe_fragments(raw)
    edges=match_fragments(ends);adj={c['component']:set() for c in components}
    for edge in edges:
        if edge['state']=='fragment_pair_candidate':
            a,b=edge['components'];adj[a].add(b);adj[b].add(a)
    hits={c['component']:{name:int(((labels==c['component'])&r).sum()) for name,r in regions.items()} for c in components}
    # Original observed components are not replaced; only a separate virtual graph.
    visited=set();paths=[]
    for start in adj:
        if start in visited:continue
        group=set();todo=[start]
        while todo:
            current=todo.pop()
            if current in group:continue
            group.add(current);todo.extend(adj[current]-group)
        visited|=group
        totals={name:sum(hits[c][name] for c in group) for name in regions}
        if all(v>=8 for v in totals.values()):
            paths.append(dict(components=sorted(group),anchor_hits=totals,
                              contains_inferred_gap=len(group)>1,physical_identity_confirmed=False))
    return dict(endpoints=ends,edges=edges,anchor_path_candidates=paths,
                native_components=components,observed_pixels_added=0,
                electrical_continuity='not_assessed',physical_identity_confirmed=False),skeleton
