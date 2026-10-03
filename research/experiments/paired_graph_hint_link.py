"""Exact rounded mapped-hint to original-native linkage; never inverse warp."""
import copy
from inspection_agent.optional_port_crop_review import aligned_predictions


def accepted_native_rows(candidates,hints,matrix,source_shape,reference_shape):
    native=copy.deepcopy(candidates)
    for row in native:row['support_tiles']=[]
    mapped=aligned_predictions(native,matrix,source_shape,reference_shape)
    accepted=[];used=set()
    for hint in hints:
        box=hint['box'];choices=[i for i,row in enumerate(mapped) if i not in used and row['class_id']==box['class_id'] and
            all(row[k]==box[k] for k in ('left','top','right','bottom')) and row['confidence']==box['confidence']]
        if len(choices)!=1:raise ValueError('ambiguous_or_unlinked_native_hint')
        i=choices[0];used.add(i);accepted.append(copy.deepcopy(candidates[i]))
    return accepted
