"""Occluder-witnessed virtual edges, never observed electrical continuity."""
from copy import deepcopy
import cv2
import numpy as np
from soft_tangent_candidates import PARAMS

OBSTACLE_PARAMS=dict(white_min_value=128,white_max_saturation=64,dark_max_value=48,
    min_area=24,max_area_fraction=.25,min_axis_ratio=2.,min_missing_pixels=3,
    min_missing_coverage=.60,max_endpoint_distance_widths=2.)


def obstacles(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);h,w=rgb.shape[:2];out=[]
    for kind,mask in [('achromatic_light',(hsv[:,:,1]<=64)&(hsv[:,:,2]>=128)),('dark',hsv[:,:,2]<=48)]:
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
        for c in range(1,n):
            area=int(stats[c,cv2.CC_STAT_AREA])
            if not 24<=area<=.25*h*w:continue
            yy,xx=np.nonzero(labels==c);points=np.column_stack((xx,yy))
            eig=np.linalg.eigvalsh(np.cov(points.T));ratio=float(np.sqrt(eig[-1]/max(eig[0],1e-9)))
            if ratio<2:continue
            out.append(dict(kind=kind,component=c,pixels=area,axis_ratio=ratio,mask=labels==c,xy=points))
    return out


def curve_pixels(curve,shape):
    canvas=np.zeros(shape,np.uint8);points=np.rint(curve).astype(np.int32)
    if (points[:,0]<0).any() or (points[:,0]>=shape[1]).any() or (points[:,1]<0).any() or (points[:,1]>=shape[0]).any():return None
    cv2.polylines(canvas,[points],False,1,1)
    return canvas>0


def witness(edge,ends,raw,objects):
    line=curve_pixels(edge['virtual_curve_xy'],raw.shape)
    if line is None:return dict(candidate=False,reason='curve_outside_image',objects=[])
    missing=line&~raw;count=int(missing.sum());found=[]
    if count<3:return dict(candidate=False,reason='too_few_missing_pixels',objects=[])
    a,b=[ends[i] for i in edge['endpoint_ids']];limit=2*max(a['width'],b['width'])
    for obj in objects:
        hits=int((missing&obj['mask']).sum());fraction=hits/count
        if fraction<.60:continue
        distances=[];outside=True
        for end in [a,b]:
            x,y=np.rint(end['point_xy']).astype(int)
            if not 0<=x<raw.shape[1] or not 0<=y<raw.shape[0] or obj['mask'][y,x]:outside=False;break
            distances.append(float(np.linalg.norm(obj['xy']-np.array([x,y]),axis=1).min()))
        if not outside or max(distances)>limit:continue
        found.append(dict(kind=obj['kind'],component=obj['component'],pixels=obj['pixels'],axis_ratio=obj['axis_ratio'],
            missing_curve_pixels=count,covered_missing_pixels=hits,coverage_fraction=fraction,endpoint_distances=distances,
            semantic_label_confirmed=False,physical_identity_confirmed=False))
    return dict(candidate=len(found)==1,reason='unique_achromatic_obstacle_candidate' if len(found)==1 else 'no_unique_obstacle',objects=found)


def augment_edges(ends,edges,rgb,raw):
    result=deepcopy(edges);objects=obstacles(rgb);by_id={e['endpoint_id']:e for e in ends}
    occupied={e for edge in edges if edge['state']=='fragment_pair_candidate' for e in edge['endpoint_ids']}
    ranking={e:[] for e in by_id}
    for i,edge in enumerate(result):
        edge['occluder_witness']=witness(edge,by_id,raw,objects)
        eligible=edge['state']!='fragment_pair_candidate' and not occupied.intersection(edge['endpoint_ids']) and edge['occluder_witness']['candidate'] and edge['score']<=PARAMS['max_score']
        if eligible:
            for endpoint in edge['endpoint_ids']:ranking[endpoint].append((edge['score'],i))
    choices={}
    for endpoint,rows in ranking.items():
        rows.sort();choices[endpoint]=rows[0][1] if rows and (len(rows)==1 or rows[1][0]-rows[0][0]>=PARAMS['margin']) else None
    for i,edge in enumerate(result):
        if edge['state']!='fragment_pair_candidate' and all(choices[e]==i for e in edge['endpoint_ids']):
            edge['state']='occluder_supported_fragment_candidate'
        edge['observed_gap_pixels']=0;edge['physical_identity_confirmed']=False
    return result
