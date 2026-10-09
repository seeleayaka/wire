"""Typed research fallback: raised lead-emergence anchors only, never sockets."""
def choose_pose(anchor,old,affine):
    if old['id']!=anchor['id'] or affine['id']!=anchor['id']:raise ValueError('anchor identity mismatch')
    if old['localization_proposal_supported']:return old
    if anchor['kind']!='visible_lead_emergence':return old
    if affine['localization_proposal_supported'] and affine.get('gates') and all(v is True for v in affine['gates'].values()):return affine
    return old
