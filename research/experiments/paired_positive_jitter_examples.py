"""Uniform training-only near-correct boxes; no photo-specific parameters."""


def positive_jitters(box):
    l,t,r,b=map(float,box);w,h=r-l,b-t;cx,cy=(l+r)/2,(t+b)/2
    if min(w,h)<=0:raise ValueError('Invalid jitter box')
    output=[]
    for dx,dy in ((-.15*w,0),(.15*w,0),(0,-.15*h),(0,.15*h)):
        output.append([l+dx,t+dy,r+dx,b+dy])
    for scale in (.8,1.2):output.append([cx-w*scale/2,cy-h*scale/2,cx+w*scale/2,cy+h*scale/2])
    return output
