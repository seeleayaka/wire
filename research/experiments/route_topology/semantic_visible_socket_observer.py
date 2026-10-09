"""Research appearance-only visible-body rescue, never a topology acceptance."""
import json
import cv2
import numpy as np
from core import sha256
from translation_socket_observer import TranslationObserver
from compound_socket_observer import ROOT

def visible_rescue(previous,labels):
    if previous not in [0,1,None] or isinstance(previous,bool):raise ValueError('typed prior label')
    if len(labels)!=9 or any(v not in [0,1,None] or isinstance(v,bool) for v in labels):raise ValueError('nine typed labels required')
    return previous if previous is not None else (1 if all(v==1 for v in labels) else None)

class SemanticVisibleObserver(TranslationObserver):
    def __init__(self):
        super().__init__()
        rp=ROOT/'artifacts/mendeley_semantic_visible_pose_rescue_20261006/report.json'
        ap=ROOT/'artifacts/mendeley_semantic_visible_pose_rescue_audit_20261006/report.json'
        r=json.loads(rp.read_text(encoding='utf-8'));a=json.loads(ap.read_text(encoding='utf-8'))
        if not r['strict_net_source_gain'] or a['status']!='PASS' or a['source_report_sha256']!=sha256(rp):raise ValueError('source/audit gate required')
        self.pins.update({str(p):sha256(p) for p in [rp,ap]})
    def infer_original(self,rgb,pose):
        result,patch=super().infer_original(rgb,pose)
        previous=result['visual_label_candidate'];labels=[]
        if previous is None:
            for x in [-2,0,2]:
                for y in [-2,0,2]:
                    h=np.array([[1,0,-1600+x],[0,1,-1000+y],[0,0,1]])@np.asarray(pose)
                    if not cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),h,(100,50),flags=cv2.INTER_NEAREST).all():raise ValueError('missing actual source pixels')
                    shifted=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
                    labels.append(super().infer(shifted)['semantic_candidate'])
            candidate=visible_rescue(previous,labels)
        else:candidate=previous
        result.update(visual_label_candidate=candidate,prior_compound_candidate=previous,
            semantic_visible_outer_labels=labels,semantic_visible_rescue=candidate==1 and previous is None,
            phenotype='mating_body_visible' if candidate==1 else 'socket_contacts_exposed' if candidate==0 else 'uncertain',
            independent_observer_count=1,new_confirmed_connections=0,electrical_continuity='not_assessed')
        return result,patch
