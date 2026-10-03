"""Train-only systematic box perturbations and GT-derived training labels."""
from audit_port_multiscale_acceptance import overlap


def perturbations(box):
    l,t,r,b=map(float,box);w,h=r-l,b-t;cx,cy=(l+r)/2,(t+b)/2
    if w<=0 or h<=0:raise ValueError('invalid GT box')
    rows=[]
    for dx,dy in ((-.75*w,0),(.75*w,0),(0,-.75*h),(0,.75*h)):
        rows.append([l+dx,t+dy,r+dx,b+dy])
    for scale in (.5,2.):rows.append([cx-w*scale/2,cy-h*scale/2,cx+w*scale/2,cy+h*scale/2])
    return rows


def training_label(box,targets):
    ranked=sorted(((overlap(box,row['box']),-row['class_id'],i) for i,row in enumerate(targets)),reverse=True)
    if not ranked:return 0,0.
    value,negative_class,_=ranked[0]
    return (1-negative_class if value>=.5 else 0),value
