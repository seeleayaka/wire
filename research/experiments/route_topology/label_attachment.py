"""Pixel-supported text-on-mask nominations, NOT wire IDs or terminal ports.

One SAM weight and one OCR weight remain single observers across all views.
No nearest-mask assignment, token repair, transitive label merging or GT use.
"""
from copy import deepcopy
from collections import defaultdict
import hashlib
import math

import cv2
import numpy as np
from identity_ocr import summarize_readings


def canonical_masks(masks, size):
    width,height=size
    if type(width) is not int or type(height) is not int or min(width,height)<=0:
        raise ValueError('positive integer frame required')
    grouped={}
    seen=set()
    for item in masks:
        identity=item['id']
        if not isinstance(identity,str) or not identity.strip() or identity in seen:
            raise ValueError('unique nonempty mask IDs required')
        seen.add(identity)
        raw=np.asarray(item['pixels'])
        if raw.shape!=(height,width) or not np.isfinite(raw).all():
            raise ValueError('mask frame/nonfinite mismatch')
        binary=np.ascontiguousarray(raw>0)
        checksum=hashlib.sha256(binary.astype(np.uint8).tobytes()).hexdigest()
        if checksum not in grouped:
            grouped[checksum]={'id':identity,'aliases':[], 'pixels':binary,
                               'topology_geometry_eligible':False}
        grouped[checksum]['aliases'].append(identity)
        # A duplicate cannot clear a failed source boundary/score guard.
        eligibility=item.get('topology_geometry_eligible',False)
        if type(eligibility) is not bool:
            raise ValueError('explicit eligibility boolean required')
        if len(grouped[checksum]['aliases'])==1:
            grouped[checksum]['topology_geometry_eligible']=eligibility
        else:
            grouped[checksum]['topology_geometry_eligible'] &= eligibility
    return list(grouped.values())


def polygon_pixels(points, size):
    width,height=size
    polygon=np.asarray(points,dtype=float)
    if polygon.shape!=(4,2) or not np.isfinite(polygon).all():
        raise ValueError('finite four-corner polygon required')
    if (polygon[:,0]<0).any() or (polygon[:,0]>=width).any() or (polygon[:,1]<0).any() or (polygon[:,1]>=height).any():
        raise ValueError('text polygon outside original frame')
    contour=polygon.astype(np.float32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour)<=0:
        raise ValueError('nondegenerate convex polygon required')
    raster=np.zeros((height,width),np.uint8)
    # Pixel-center raster, OpenCV fixed-point subpixel coordinates. No bbox fill.
    fixed=np.rint(polygon*256).astype(np.int32)
    cv2.fillConvexPoly(raster,fixed,1,shift=8)
    if not raster.any():
        raise ValueError('polygon has no raster pixels')
    return raster.astype(bool)


def attach_polygon(points, grouped, size):
    footprint=polygon_pixels(points,size)
    area=int(footprint.sum())
    supports=[]
    for item in grouped:
        count=int(np.logical_and(footprint,item['pixels']).sum())
        if count:
            supports.append({'mask_id':item['id'],'aliases':item['aliases'],
                             'text_polygon_fraction_on_mask':count/area,
                             'intersection_pixels':count,
                             'topology_geometry_eligible':item['topology_geometry_eligible']})
    if len(supports)>1:
        state='ambiguous_multiple_masks'
    elif not supports:
        state='no_mask_overlap'
    elif supports[0]['text_polygon_fraction_on_mask']<.5:
        state='insufficient_mask_overlap'
    else:
        state='unique_pixel_supported_nomination'
    return {'state':state,'supports':supports,
            'nominated_mask_id':supports[0]['mask_id'] if state=='unique_pixel_supported_nomination' else None,
            'confirmed_wire_identity':None,'confirmed_port_identity':None}


def nominate(readings,masks,size):
    rows=deepcopy(readings)
    grouped=canonical_masks(masks,size)
    ids=[r['record_id'] for r in rows]
    if any(not isinstance(i,str) or not i.strip() for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('unique nonempty reading IDs required')
    attachments={}
    for row in rows:
        points=np.asarray(row['polygon_source_xy'],float)
        expected=[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
        if not np.allclose(row['bbox_xyxy'],expected,rtol=0,atol=1e-8):
            raise ValueError('reading polygon/bbox mismatch')
        attachments[row['record_id']]=attach_polygon(points,grouped,size)
    texts=summarize_readings(rows)
    nominations=[]
    for group in texts['groups']:
        members=[attachments[i] for i in group['record_ids']]
        choices={m['nominated_mask_id'] for m in members}
        valid=len(choices)==1 and None not in choices
        nomination={'text_group':group,'state':'unique_text_on_visible_mask' if valid else 'unresolved_text_membership',
                    'nominated_mask_id':next(iter(choices)) if valid else None,
                    'all_reading_memberships':members,
                    'high_score_consistent_text_on_mask':bool(valid and group['ocr_high_score_consistent']),
                    'sam_model_observer_count':1,'ocr_model_observer_count':1,
                    'confirmed':False,'wire_identity':None,'terminal_assignment':None,
                    'automatic_connections':[]}
        nominations.append(nomination)
    per_text=defaultdict(set)
    for item in nominations:
        if item['high_score_consistent_text_on_mask']:
            per_text[item['text_group']['best_text']].add(item['nominated_mask_id'])
    collisions={text:sorted(mask_ids) for text,mask_ids in per_text.items() if len(mask_ids)>1}
    for item in nominations:
        item['same_text_on_distinct_masks']=item['text_group']['best_text'] in collisions
        item['text_identity_usable_automatically']=False
    return {'groups':nominations,'reading_memberships':attachments,
            'same_text_different_masks':collisions,'canonical_mask_count':len(grouped),
            'original_mask_count':len(masks),'automatic_connections':[],
            'confirmed_ports':[],'new_confirmed_connections':0,
            'claim_boundary':'Pixel-supported OCR nomination only. A SAM mask may be a sleeve, tool or partial wire. Text is not authenticated wire/terminal identity, expected wiring or continuity.'}


def mask_context_crop(pixels,size):
    width,height=size
    raw=np.asarray(pixels)
    if raw.shape!=(height,width) or not np.isfinite(raw).all():
        raise ValueError('mask frame mismatch')
    ys,xs=np.where(raw>0)
    if not len(xs):
        return None
    box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
    padding=max(4,math.ceil(.5*min(box[2]-box[0],box[3]-box[1])))
    return [max(0,box[0]-padding),max(0,box[1]-padding),
            min(width,box[2]+padding),min(height,box[3]+padding)]
