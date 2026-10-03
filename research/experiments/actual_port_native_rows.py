"""Forward-link actual displayed cues; never inverse-warp rounded hint boxes."""
from paired_graph_hint_link import accepted_native_rows

def actual_native_rows(report,result):
    import numpy as np
    hints=result.get('rescue_hints',[])+result.get('supplementary_hints',[])
    if not hints:return []
    pool=[]
    def collect(value):
        if isinstance(value,dict):
            if all(k in value for k in ('box_xyxy','confidence','class_id')):pool.append(value)
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    source=result['source_evidence'];collect(source['predictions'])
    from inspection_agent.context_port_recheck import confirm_rechecks
    views=result.get('recheck_evidence',{}).get('source_views',[])
    if views:collect(confirm_rechecks(views,source['predictions']['source_shape']))
    for name,keys in (
        ('teacher_student_evidence',('teacher_case','student_case','fusion')),
        ('feature_residual_evidence',('feature_case','fusion')),
        ('resolution_evidence',('alternative','fusion')),
        ('paired_geometry_evidence',('native_current','native')),
        ('median_geometry_evidence',('native_current','native')),
        ('native_pose_evidence',('native_current','native')),
        ('pose_geometry_evidence',('native_current','native'))):
        evidence=result.get(name,{})
        for key in keys:collect(evidence.get(key,{}))
    unique={}
    for row in pool:
        key=(row['class_id'],row['confidence'],*row['box_xyxy']);unique.setdefault(key,row)
    shape=source['predictions']['source_shape']
    return accepted_native_rows(list(unique.values()),hints,
        np.asarray(report['alignment']['source_to_reference_homography']),shape,shape)
