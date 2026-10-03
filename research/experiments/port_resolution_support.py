"""Resolution-only residual experiment, same checkpoint and strict selectors."""
import copy
from core_port_precision_policy import iou
from teacher_student_port_policy import merge

class ResolutionModel:
    def __init__(self,model,resolution,restore_threads):
        if resolution not in (960,1280):raise ValueError('unfrozen input resolution')
        self.model=model;self.resolution=resolution;self.restore_threads=restore_threads
    def predict(self,*args,**kwargs):
        self.restore_threads();kwargs['imgsz']=self.resolution
        output=self.model.predict(*args,**kwargs);self.restore_threads();return output

def append_resolution(teacher,current,alternative):
    output=copy.deepcopy(current);output.update(resolution_additions=[],resolution_fallback_reason=None)
    additional=merge(teacher,alternative)
    if additional['fallback_reason']:
        output['resolution_fallback_reason']=additional['fallback_reason'];return output
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    for row in additional['student_additions']:
        if len(output['resolution_additions'])>=remaining:break
        if any(iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in output['all_predictions']):continue
        row=copy.deepcopy(row);row.update(evidence_tier='resolution_residual_manual_review',inference_imgsz=1280)
        output['resolution_additions'].append(row);output['all_predictions'].append(row)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(output['all_predictions'])<=len(output['primary'])+5 and len(output['primary'])<=5
    return output
