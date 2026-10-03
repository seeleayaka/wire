"""Frozen area-weighted candidate footprint pooling, no metadata channels."""
import torch
import torch.nn.functional as F
from port_semantic_verifier import crop_tensor,CONTEXT_SCALES
from paired_shape_features import crop_signature


def footprint_weights(box,signature,grid=16):
    l,t,r,b=map(float,box);left,top,right,bottom=signature
    if min(r-l,b-t,right-left,bottom-top)<=0:raise ValueError('Invalid footprint geometry')
    x=torch.linspace(float(left),float(right),grid+1);y=torch.linspace(float(top),float(bottom),grid+1)
    horizontal=(torch.minimum(x[1:],torch.tensor(r))-torch.maximum(x[:-1],torch.tensor(l))).clamp(min=0)
    vertical=(torch.minimum(y[1:],torch.tensor(b))-torch.maximum(y[:-1],torch.tensor(t))).clamp(min=0)
    weights=vertical[:,None]*horizontal[None,:]
    if not torch.isfinite(weights).all() or weights.sum()<=0:raise ValueError('Empty candidate footprint')
    return weights/weights.sum()


def embeddings(model,image,boxes,*,audit=None):
    unique={};tensors=[];ordered=[]
    for box in boxes:
        indices=[]
        for scale in CONTEXT_SCALES:
            signature=crop_signature(box,scale)
            if signature not in unique:
                unique[signature]=len(tensors);tensors.append(crop_tensor(image,box,scale))
            indices.append((unique[signature],signature))
        ordered.append(indices)
    cls_rows=[];patch_rows=[]
    with torch.inference_mode():
        for start in range(0,len(tensors),8):
            result=model.forward_features(torch.stack(tensors[start:start+8]))
            cls_rows.append(F.normalize(result['x_norm_clstoken'],dim=1))
            patch_rows.append(result['x_norm_patchtokens'].reshape(-1,16,16,384))
    if audit is not None:audit.update(boxes=len(boxes),requested_crops=len(boxes)*2,unique_crops=len(tensors))
    if not boxes:return torch.empty((0,1536))
    classes=torch.cat(cls_rows);patches=torch.cat(patch_rows);vectors=[]
    for box,records in zip(boxes,ordered):
        scales=[]
        for index,signature in records:
            weights=footprint_weights(box,signature).to(patches.dtype)
            pooled=F.normalize((patches[index]*weights[:,:,None]).sum(dim=(0,1)),dim=0)
            scales.append(torch.cat((classes[index],pooled)))
        vectors.append(torch.cat(scales)/2.)
    output=torch.stack(vectors)
    if output.shape!=(len(boxes),1536) or not torch.isfinite(output).all():raise ValueError('Invalid footprint embedding')
    return output
