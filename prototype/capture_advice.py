"""Deterministic local evidence-collection guidance. No image/API/model calls.

Suggestions do not identify occlusion, certify electrical continuity, or alter
detector boxes. Supplemental views must not silently replace same-view pairs.
"""
from __future__ import annotations
from typing import Any

def build_capture_advice(report: dict[str,Any]) -> dict[str,Any]:
    alignment=report.get('alignment') or {}
    qualities=[q for q in [report.get('alignment_quality'),alignment.get('alignment_quality')]
               if isinstance(q,dict)]
    flags=[q.get('reliable') for q in qualities]
    state='unreliable' if any(v is False for v in flags) else ('reliable' if any(v is True for v in flags) else 'unknown')
    if report.get('decision')=='alignment_uncertain_manual_review':state='unreliable'
    regions=report.get('review_regions') or []
    if not isinstance(regions,list):regions=[]
    ids=[f'candidate_{i:03d}' for i in range(1,len(regions)+1)]
    items=[]
    def add(code,text,candidates=None):
        items.append({'code':code,'text':text,'candidate_ids':candidates or []})
    if state=='unreliable':
        add('retake_same_view','图像定位不可靠：先按参考图的正面角度、距离和取景范围重拍，保持设备边缘与端子排完整，不用新的侧视图直接替代对比图。')
    elif state=='unknown':
        add('retake_same_view','当前没有可靠的定位质量结论：对比照片应尽量保持与参考图相同的角度、距离和取景范围。')
    else:
        add('same_view_baseline','对比照片保持参考图的角度、距离和取景范围；定位通过不代表没有局部视差。')
    if ids:
        add('closeup_candidates',f'对当前{len(ids)}个复核框补拍清晰局部：先同角度靠近或放大，保留线材、接头及周边定位参照，避免只拍框内一小块。',ids)
        add('supplemental_angle','如标签、线束或接头挡住目标，可从安全的左侧或右侧补一个能看清遮挡处的角度；这是补充证据，不能直接当成同角度对比图，也不能凭照片确认导通。',ids)
    fusion=report.get('sam3_fusion') or {}
    if isinstance(fusion,dict) and fusion.get('status')=='error':
        add('sam_unavailable','SAM线材证据本轮不可用：保留已有复核框，回看清晰原图；必要时补拍局部，不把掩膜缺失视为实物缺线。',ids)
    if not ids:
        add('no_candidate_boundary','当前未列出复核框不等于电气连接已经验证；如仍有可疑位置，可另拍局部供人工核查。')
    add('focus_lighting','拍摄时固定相机、对焦到端子和线材，避免反光、阴影及手抖；先检查照片能否看清线材边缘。')
    add('capture_safety','只在现场许可的安全位置拍摄，不移动带电线材、不拆标签、不触碰接头；需要实物操作或导通检查时交由具备资质人员。')
    return {'schema_version':1,'source':'local_capture_rules','requires_llm':False,'network_requests':0,
            'alignment_state':state,'comparison_view':'same_as_reference',
            'supplemental_angle_is_comparison_image':False,'candidate_count':len(ids),
            'items':items,'occlusion_confirmed':False,'electrical_verdict_assessed':False,
            'detector_decision_unchanged':True}

def render_capture_advice(advice: dict[str,Any]) -> str:
    return '拍摄建议（本地生成，无需大模型）\n'+'\n'.join(
        f'{index}. {item["text"]}' for index,item in enumerate(advice['items'],1))
