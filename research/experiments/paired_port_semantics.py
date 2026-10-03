"""GT-free paired features and geometry primitives for conditional crop review."""
import math
import cv2
import numpy as np
import torch
from port_semantic_verifier import embeddings,context_box,CONTEXT_SCALES

MIN_VALID_COVERAGE=.85


def paired_features(observed,expected):
    if observed.ndim!=2 or observed.shape!=expected.shape or observed.shape[1]!=1536:
        raise ValueError('paired embedding shape mismatch')
    if not torch.isfinite(observed).all() or not torch.isfinite(expected).all():raise ValueError('nonfinite paired embeddings')
    return torch.cat((observed,expected,(observed-expected).abs(),observed*expected),dim=1)


def expected_in_source(reference,matrix,shape):
    h,w=shape;matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)!=3 or min(h,w)<1:
        raise ValueError('invalid reference alignment')
    inverse=np.linalg.inv(matrix)
    expected=cv2.warpPerspective(reference,inverse,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=cv2.warpPerspective(np.full(reference.shape[:2],255,np.uint8),inverse,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    valid=cv2.erode(valid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
    return expected,valid


def context_valid_fraction(box,mask,scale):
    l,t,r,b=context_box(box,scale);l,t,r,b=math.floor(l),math.floor(t),math.ceil(r),math.ceil(b)
    h,w=mask.shape;part=mask[max(0,t):min(h,b),max(0,l):min(w,r)] if r>0 and b>0 and l<w and t<h else np.zeros((0,0),bool)
    return float(part.sum())/max(1,(r-l)*(b-t))


def valid_boxes(boxes,mask):
    return [i for i,box in enumerate(boxes) if all(context_valid_fraction(box,mask,s)>=MIN_VALID_COVERAGE for s in CONTEXT_SCALES)]


class CachedReferenceSIFT:
    """Original detector parameters/matcher/gates, immutable reference cache."""
    def __init__(self,reference):
        self.gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY);self.original=cv2.SIFT_create;self.cached=None;self.cache_hits=0
    def __enter__(self):
        owner=self
        def create(*a,**kw):
            native=owner.original(*a,**kw)
            if a or kw!={'nfeatures':9000,'contrastThreshold':.014,'edgeThreshold':12}:return native
            class Proxy:
                def detectAndCompute(self,gray,mask):
                    if mask is None and gray.shape==owner.gray.shape and np.array_equal(gray,owner.gray):
                        if owner.cached is None:owner.cached=native.detectAndCompute(gray,mask)
                        else:owner.cache_hits+=1
                        return owner.cached
                    return native.detectAndCompute(gray,mask)
            return Proxy()
        cv2.SIFT_create=create;return self
    def __exit__(self,*exc):cv2.SIFT_create=self.original
