"""Source-group curation, training-only labels; no runtime image rules."""
import math
from inspection_agent.port_tiling import box_iou


def native_label(proposal, targets):
    return proposal['class_id'] + 1 if any(
        proposal['class_id'] == target['class_id'] and
        box_iou(proposal['box_xyxy'], target['box']) >= .5
        for target in targets) else 0


def pixel_signature(record):
    l,t,r,b = map(float, record['box'])
    if not all(math.isfinite(v) for v in (l,t,r,b)) or r <= l or b <= t:
        raise ValueError('Invalid training crop')
    cx,cy = (l+r)/2,(t+b)/2
    contexts = []
    for scale in (1.5,3.0):
        side = max(24.,max(r-l,b-t)*scale)
        contexts.append((math.floor(cx-side/2),math.floor(cy-side/2),
                         math.ceil(cx+side/2),math.ceil(cy+side/2)))
    reference = record.get('observed_is_expected_reference',False) or record['kind'] in ('reference_self','jitter_reference_self')
    return record['image'], bool(reference), tuple(contexts)


def curate(records, source_folds):
    grouped = {}
    for index,row in enumerate(records):
        if row['image'] not in source_folds or row['fold'] != source_folds[row['image']]:
            raise ValueError('Source fold mismatch')
        if row['label'] not in (0,1,2): raise ValueError('Invalid class')
        grouped.setdefault(pixel_signature(row),[]).append(index)
    kept,duplicates,conflicts = [],[],[]
    for indices in grouped.values():
        if len({records[i]['label'] for i in indices}) > 1:
            conflicts.extend(indices)
        else:
            kept.append(indices[0]); duplicates.extend((indices[0],i) for i in indices[1:])
    return sorted(kept),duplicates,sorted(conflicts)
