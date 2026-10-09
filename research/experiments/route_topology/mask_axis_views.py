"""Whole-mask PCA orientation for a bounded diagnostic view, not identity."""
import math
import cv2
import numpy as np


def axis_transform(pixels,crop):
    raw=np.asarray(pixels)
    if raw.ndim!=2 or not np.isfinite(raw).all():raise ValueError('finite HW mask required')
    if len(crop)!=4 or any(type(v) is not int for v in crop):raise ValueError('integer crop required')
    x0,y0,x1,y1=crop;h,w=raw.shape
    if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('crop outside frame')
    yy,xx=np.where(raw>0)
    if len(xx)<2:return dict(state='insufficient_mask_pixels')
    if xx.min()<x0 or xx.max()>=x1 or yy.min()<y0 or yy.max()>=y1:
        raise ValueError('whole-mask support must fit crop')
    xy=np.column_stack((xx-x0,yy-y0)).astype(float)
    centered=xy-xy.mean(0);cov=centered.T@centered/len(xy)
    values,vectors=np.linalg.eigh(cov);small,big=map(float,values)
    if big<=1e-9 or big<2*max(0.,small):return dict(state='orientation_ambiguous',eigenvalues=values.tolist())
    direction=vectors[:,-1]
    if direction[0]<0 or (abs(direction[0])<1e-12 and direction[1]<0):direction=-direction
    vx,vy=map(float,direction);rotation=np.array([[vx,vy],[-vy,vx]])
    corners=np.array([[0,0],[x1-x0-1,0],[x1-x0-1,y1-y0-1],[0,y1-y0-1]],float)@rotation.T
    lo=corners.min(0);hi=corners.max(0)
    matrix=np.column_stack((rotation,2-lo))
    size=[int(math.ceil(hi[0]-lo[0]))+5,int(math.ceil(hi[1]-lo[1]))+5]
    return dict(state='oriented_diagnostic_view',source_crop_xyxy=list(crop),output_size=size,
                source_crop_to_oriented=matrix.tolist(),oriented_to_source_crop=cv2.invertAffineTransform(matrix).tolist(),
                principal_direction=direction.tolist(),eigenvalues=values.tolist(),
                minimum_anisotropy=2.,automatic_wire_identity=None,observer_count=1)


def warp_context(image,crop,transform):
    if transform['state']!='oriented_diagnostic_view':raise ValueError('orientation unavailable')
    x0,y0,x1,y1=crop
    if image.ndim!=3 or image.shape[2]!=3:raise ValueError('RGB image required')
    matrix=np.asarray(transform['source_crop_to_oriented'],float)
    return cv2.warpAffine(image[y0:y1,x0:x1],matrix,tuple(transform['output_size']),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))


def map_to_source(points,transform):
    p=np.asarray(points,float)
    if p.shape!=(4,2) or not np.isfinite(p).all():raise ValueError('finite OCR quadrilateral required')
    matrix=np.asarray(transform['oriented_to_source_crop'],float)
    original=p@matrix[:,:2].T+matrix[:,2]
    return original+np.asarray(transform['source_crop_xyxy'][:2])
