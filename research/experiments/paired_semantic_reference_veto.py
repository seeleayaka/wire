"""GT-free ORIGINAL strong-reference same-class IoU veto, no geometry edits."""
import copy
from inspection_agent.optional_port_crop_review import aligned_predictions
from inspection_agent.port_tiling import box_iou


def veto(candidates, reference_rows, matrix, source_shape, reference_shape):
    native=copy.deepcopy(candidates)
    for row in native:row['support_tiles']=[]
    mapped=aligned_predictions(native,matrix,source_shape,reference_shape)
    if len(mapped)!=len(native):raise ValueError('unexpected mapped candidate count')
    coords=lambda row:[row[k] for k in ('left','top','right','bottom')]
    strong=[row for row in reference_rows if row['confidence']>.25]
    kept=[];audits=[]
    for original,row in zip(candidates,mapped):
        overlaps=[box_iou(coords(row),coords(ref)) for ref in strong if ref['class_id']==row['class_id']]
        maximum=max(overlaps,default=0.)
        rejected=maximum>=.5
        audits.append(dict(box_xyxy=original['box_xyxy'],class_id=original['class_id'],
                           strong_reference_maximum_same_class_iou=maximum,rejected=rejected))
        if not rejected:kept.append(copy.deepcopy(original))
    return kept,audits
