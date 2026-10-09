"""Only penalize mismatch; never lower a geometry score to force a match."""
import cv2
import numpy as np
from soft_tangent_candidates import direction_evidence,geometry as soft_geometry,PARAMS
from tangent_gap_candidates import describe_fragments


def color_profile(endpoint,raw,labels,rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10
    selection=np.zeros(raw.shape,bool);radius=max(1,int(round(endpoint['width']/2)))
    h,w=raw.shape
    for x,y in endpoint['support_points_xy']:
        x,y=int(x),int(y);selection[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]=True
    selection &= raw&(labels==endpoint['component'])
    values=bins[selection]
    if not len(values):return None
    hist=np.bincount(values,minlength=18).astype(float);return (hist/hist.sum()).tolist()


def geometry(a,b):
    edge=soft_geometry(a,b)
    if edge is None or a.get('color_histogram') is None or b.get('color_histogram') is None:return None
    h=np.asarray(a['color_histogram']);k=np.asarray(b['color_histogram'])
    distance=float(np.sqrt(max(0.,1-float(np.sqrt(h*k).sum()))))
    edge.update(base_geometry_score=edge['score'],color_distance=distance,color_penalty=.25*distance)
    edge['score']+=edge['color_penalty'];return edge


def match(ends):
    edges=[];rank={e['endpoint_id']:[] for e in ends}
    for i,a in enumerate(ends):
        for b in ends[i+1:]:
            edge=geometry(a,b)
            if edge is None:continue
            index=len(edges);edges.append(edge)
            for e in edge['endpoint_ids']:rank[e].append((edge['score'],index))
    choices={}
    for endpoint,rows in rank.items():
        rows.sort();choices[endpoint]=rows[0][1] if rows and rows[0][0]<=PARAMS['max_score'] and (len(rows)==1 or rows[1][0]-rows[0][0]>=PARAMS['margin']) else None
    for i,e in enumerate(edges):e['state']='fragment_pair_candidate' if all(choices[p]==i for p in e['endpoint_ids']) else 'ambiguous_nonreciprocal_or_low_score'
    return edges


def observe_family(raw,regions,rgb):
    labels,oldends,components,skel=describe_fragments(raw);ends=[]
    for e in oldends:
        e=direction_evidence(e);e['color_histogram']=color_profile(e,raw,labels,rgb);ends.append(e)
    edges=match(ends);adj={c['component']:set() for c in components}
    for e in edges:
        if e['state']=='fragment_pair_candidate':
            a,b=e['components'];adj[a].add(b);adj[b].add(a)
    paths=[];seen=set()
    for c in adj:
        if c in seen:continue
        todo=[c];group=set()
        while todo:
            q=todo.pop()
            if q in group:continue
            group.add(q);todo.extend(adj[q]-group)
        seen|=group;selection=np.isin(labels,list(group));hits={k:int((selection&r).sum()) for k,r in regions.items()}
        if all(v>=8 for v in hits.values()):paths.append(dict(components=sorted(group),anchor_hits=hits,contains_inferred_gap=len(group)>1,physical_identity_confirmed=False))
    return dict(endpoints=ends,edges=edges,anchor_path_candidates=paths,native_components=components,
        raw_anchor_hits={k:int((raw&r).sum()) for k,r in regions.items()},observed_pixels_added=0,
        physical_identity_confirmed=False,electrical_continuity='not_assessed'),skel
