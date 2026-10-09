"""Bounded alignment nuisance is ONE contact cue, never independent votes."""
import json
import cv2
import numpy as np
from compound_socket_observer import Observer, ROOT, resolve
from run_positive_contact_gate import bright_contacts
from core import sha256

class TranslationObserver(Observer):
    def __init__(self):
        super().__init__()
        report_path=ROOT/'artifacts/mendeley_contact_translation_nuisance_20261006/report.json'
        audit_path=ROOT/'artifacts/mendeley_contact_nuisance_audit_20261006/report.json'
        report=json.loads(report_path.read_text(encoding='utf-8'));audit=json.loads(audit_path.read_text(encoding='utf-8'))
        if not report['strict_robust_source_gain'] or audit['status']!='PASS' or audit['report_sha256']!=sha256(report_path):raise ValueError('robust source gate/audit required')
        self.pins.update({str(p):sha256(p) for p in [report_path,audit_path]})
        self.threshold=report['threshold']
    def infer_original(self,rgb,pose):
        matrix=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.asarray(pose)
        patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
        result=super().infer(patch)
        grid={}
        for x in [-4,-2,0,2,4]:
            for y in [-4,-2,0,2,4]:
                h=np.array([[1,0,-1600+x],[0,1,-1000+y],[0,0,1]])@np.asarray(pose)
                valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),h,(100,50),flags=cv2.INTER_NEAREST)
                if not valid.all():raise ValueError('nuisance patch lacks actual original pixels')
                shifted=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
                grid[(x,y)]=float(bright_contacts(shifted)[self.support].mean())
        offsets=[(x,y) for x in [-2,0,2] for y in [-2,0,2]]
        features=[max(grid[(x+dx,y+dy)] for dx,dy in offsets) for x,y in offsets]
        positive=min(features)>self.threshold
        candidate=resolve(result['old_candidate'],result['color_candidate'],result['semantic_candidate'],positive)
        result.update(visual_label_candidate=candidate,contact_score=features[4],contact_outer_scores=features,
            positive_contact_cue=positive,contact_nuisance_threshold=self.threshold,
            phenotype='mating_body_visible' if candidate==1 else 'socket_contacts_exposed' if candidate==0 else 'uncertain',
            one_image_translation_probes_not_independent_votes=True,pose_robust_contact_required_for_new_rescue=True)
        return result,patch
