"""List competing visible footprints, not inferred wires or electrical edges."""
import hashlib
import math
import numpy as np


def assess_competition(records,endpoint_ids):
    if len(endpoint_ids)!=len(set(endpoint_ids)):
        raise ValueError('duplicate endpoint ID')
    seen=set();shapes=set();groups={identity:{} for identity in endpoint_ids}
    for row in records:
        if row['record_id'] in seen:
            raise ValueError('duplicate native record ID; must not count twice')
        seen.add(row['record_id'])
        score=row['score']
        if not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('invalid native score')
        for identity,data in row['endpoints'].items():
            if identity not in groups:
                raise ValueError('unbound endpoint ID')
            mask=np.asarray(data['matched_pixels'])
            if mask.dtype!=bool or mask.ndim!=2:
                raise ValueError('two-dimensional boolean local footprint required')
            shapes.add(mask.shape)
            if len(shapes)>1:
                raise ValueError('native coordinate frames/shapes differ')
            if score<.75 or data['appearance_state']!='reference_color_supported' or not mask.any():
                continue
            fingerprint=hashlib.sha256(str(mask.shape).encode()+mask.tobytes()).hexdigest()
            # Exact same footprint only: alias listing, NOT mask union or extra model vote.
            group=groups[identity].setdefault(fingerprint,dict(footprint_sha256=fingerprint,
                native_record_ids=[],colored_pixels=int(mask.sum())))
            group['native_record_ids'].append(row['record_id'])
    endpoints={}
    for identity,footprints in groups.items():
        items=sorted(footprints.values(),key=lambda g:g['footprint_sha256'])
        for item in items:item['native_record_ids'].sort()
        state=('no_supported_local_footprint' if not items else
               'single_supported_local_footprint' if len(items)==1 else 'competing_visible_local_footprints')
        endpoints[identity]=dict(state=state,distinct_footprints=len(items),footprints=items)
    return dict(endpoints=endpoints,review_needed=any(e['distinct_footprints']>1 for e in endpoints.values()),
        same_physical_wire_confirmed=False,electrical_continuity='not_assessed',
        independent_model_observer_count=1,nearby_footprint_is_not_terminal_attachment=True,
        different_masks_are_not_proven_different_wires=True,mask_pixels_changed=False)
