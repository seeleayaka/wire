"""Non-actuating review actions; generation is not a completed physical recheck."""

DECISIONS={'same_visible_bundle_attachment_supported','visible_socket_attachment_change_supported','insufficient_evidence'}


def plan(comparison):
    decision=comparison.get('decision')
    if decision not in DECISIONS or comparison.get('observation_granularity')!='multiwire_bundle_only':
        raise ValueError('typed visible multiwire bundle comparison required')
    if comparison.get('physical_new_connections')!=0 or comparison.get('electrical_correctness')!='not_assessed':
        raise ValueError('cannot generate plan from an overstated electrical result')
    if decision=='same_visible_bundle_attachment_supported':
        action={'kind':'record_supported_visible_bundle_scope','priority':'routine',
            'message':'可见线束接法与参考一致；仅记录这组外观证据，不代替逐芯或电气检查。'}
    elif decision=='visible_socket_attachment_change_supported':
        action={'kind':'request_human_socket_reinspection','priority':'attention',
            'message':'该参考插座的接点露出，建议人工核查插接并补充近照；未判定电气断路。'}
    else:
        action={'kind':'request_additional_visible_evidence','priority':'evidence_needed',
            'message':'当前证据不足。补充能看清插头、出线处与遮挡区域的照片，再次复核；不猜接法。'}
    return {'action':action,'recheck_execution_status':'not_performed',
        'physical_actuation_requested':False,'automatic_repair_performed':False,
        'electrical_measurement_performed':False,'original_visual_review_retained':True}
