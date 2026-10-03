"""Bounded complementary cue fusion, not independent physical verification."""
import copy,math
from core_port_recheck_policy import select_zoom
from core_port_supplement_policy import supported_tiles,complete
from core_port_precision_policy import iou

POLICY=dict(maximum_primary=5,maximum_supplementary_including_recheck=5,
    student_minimum_score=.75,teacher_support_score=.25,teacher_support_iou=.5,
    deduplication_iou=.5,minimum_supporting_tiles=2,
    variants=['cross_model_supported','student_consistent'],
    preserve_teacher_all_selected=True,
    warning='Fine-tuned models and two crops of one image are correlated, not physical verification')

def valid(row):
    box=row.get('box_xyxy',[]);score=row.get('confidence')
    return (row.get('class_id') in (0,1) and len(box)==4 and
        all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) and
        0<=box[0]<box[2] and 0<=box[1]<box[3] and
        isinstance(score,(int,float)) and math.isfinite(score) and 0<=score<=1)

def merge(teacher,student,mode='cross_model_supported'):
    if mode not in POLICY['variants']:raise ValueError('unknown fusion mode')
    base=select_zoom(teacher,'strict')
    result=copy.deepcopy(base);result.update(student_additions=[],fallback_reason=None)
    if any(teacher.get(key)!=student.get(key) for key in ('image','source_sha256')) or not teacher.get('source_sha256'):
        result['fallback_reason']='source_identity_mismatch';return result
    student_raw=student.get('predictions')
    if not isinstance(student_raw,dict) or teacher['predictions']['source_shape']!=student_raw.get('source_shape'):
        result['fallback_reason']='source_geometry_mismatch';return result
    try:
        selected=select_zoom(student,'strict')
        remaining=5-len(base['supplementary'])-len(base['zoom'])
        ordered=sorted(selected['all_predictions'],key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id']))
        for row in ordered:
            if len(result['student_additions'])>=remaining:break
            if not valid(row) or row['confidence']<=.75 or not complete(row,teacher['predictions']['source_shape']):continue
            if any(iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in result['all_predictions']):continue
            votes=supported_tiles(row,student['predictions'])
            double_recheck=(row in selected['zoom'] and len(row.get('zoom_view_scores',[]))==2
                and min(row['zoom_view_scores'])>.75 and len(row.get('zoom_windows',[]))==2
                and row['zoom_windows'][0]!=row['zoom_windows'][1])
            if len(votes)<2 and not double_recheck:continue
            support=[p for p in teacher['predictions']['merged_predictions'] if valid(p) and
                p['class_id']==row['class_id'] and p['confidence']>.25 and iou(p['box_xyxy'],row['box_xyxy'])>=.5]
            if mode=='cross_model_supported' and not support:continue
            addition=copy.deepcopy(row);addition.update(evidence_tier='student_complement_manual_review',
                student_support_tiles=votes,student_double_recheck=double_recheck,
                teacher_support_maximum_score=max((p['confidence'] for p in support),default=None),
                teacher_weight_sha256=teacher.get('weight_sha256'),student_weight_sha256=student.get('weight_sha256'))
            result['student_additions'].append(addition);result['all_predictions'].append(addition)
        assert result['primary']==base['primary'] and result['supplementary']==base['supplementary'] and result['zoom']==base['zoom']
        assert len(result['primary'])<=5 and len(result['all_predictions'])<=10
        return result
    except (KeyError,TypeError,ValueError,AssertionError) as exc:
        result=copy.deepcopy(base);result.update(student_additions=[],fallback_reason=type(exc).__name__)
        return result
