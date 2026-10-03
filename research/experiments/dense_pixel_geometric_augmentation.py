"""Training-only consistent square-grid transforms; native GT re-encoded."""
import torch
from dense_pixel_port_probe import encode_targets


def transform_item(item, rotation, mirror):
    if rotation not in (0,1,2,3) or type(mirror) is not bool:
        raise ValueError('invalid dihedral transform')
    height,width=item['shape']
    if min(height,width)<=0:
        raise ValueError('invalid native shape')
    boxes=[]
    for row in item['boxes']:
        l,t,r,b=row['box']
        points=[(l/width,t/height),(r/width,t/height),(l/width,b/height),(r/width,b/height)]
        for _ in range(rotation):
            points=[(y,1-x) for x,y in points]
        if mirror:
            points=[(1-x,y) for x,y in points]
        nh,nw=(width,height) if rotation%2 else (height,width)
        xs,ys=zip(*points)
        boxes.append(dict(class_id=row['class_id'],box=[min(xs)*nw,min(ys)*nh,max(xs)*nw,max(ys)*nh]))
    nh,nw=(width,height) if rotation%2 else (height,width)
    features={}
    for key,tensor in item['features'].items():
        if tensor.ndim!=3 or tensor.shape[-2]!=tensor.shape[-1]:
            raise ValueError('square cached feature map required')
        value=torch.rot90(tensor,rotation,dims=(-2,-1))
        features[key]=(torch.flip(value,dims=(-1,)) if mirror else value).contiguous()
    encoded=encode_targets(boxes,nw,nh)
    return dict(features=features,targets={k:encoded[k] for k in ('heat','reg','mask')},shape=[nh,nw],boxes=boxes)
