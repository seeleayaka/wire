"""Conservative appearance-only rescue: never replace an old supported label.

Compound policy motivated by source component failures; NOT independent votes or
electrical edges. Source development audit must pass before live rescue is enabled.
"""
def rescue(baseline,color,semantic,source_ready,pose_ready):
    old=baseline['visual_label_candidate']
    result=dict(baseline,old_supported_label_preserved=True,new_confirmed_connections=0,
        electrical_continuity='not_assessed',independent_observer_count=1,
        component_rescue_applied=False)
    if old is not None:return result
    if not source_ready or not pose_ready:return result
    c=color['visual_label_candidate'];s=semantic['visual_label_candidate']
    if c is not None and c==s:
        result.update(visual_label_candidate=c,component_rescue_applied=True,
            rescue_scope='registered_socket_visual_phenotype_only',decision='insufficient_evidence')
    return result
