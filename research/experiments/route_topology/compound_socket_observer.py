"""Frozen appearance observer; never establishes wire identity or continuity."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from core import sha256
from component_cues import color_descriptor, semantic_descriptor, prediction
from socket_appearance import descriptor
from socket_phenotype import predict
from socket_phenotype_agreement import feature_view, agreement
from run_positive_contact_gate import bright_contacts

ROOT=Path(__file__).resolve().parents[2]
def resolve(old, color, semantic, positive):
    if any(v not in (None,0,1) or isinstance(v,bool) for v in [old,color,semantic]):
        raise ValueError('typed appearance labels required')
    if type(positive) is not bool:raise ValueError('typed contact support required')
    return old if old is not None else (0 if color==0 and semantic==0 and positive else None)

class Observer:
    def __init__(self):
        self.pins={}
        def read(path):
            self.pins[str(path)]=sha256(path)
            return json.loads(path.read_text(encoding='utf-8'))
        self.raw_old=read(ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005/model.json')
        self.raw_color=read(ROOT/'artifacts/mendeley_component_color_source_20261006/model.json')
        self.raw_semantic=read(ROOT/'artifacts/mendeley_component_semantic_source_v2_20261006/model.json')
        source=read(ROOT/'artifacts/mendeley_positive_contact_source_20261006/report.json')
        audit=read(ROOT/'artifacts/mendeley_positive_contact_audit_20261006/report.json')
        controls=read(ROOT/'artifacts/mendeley_compound_fresh_controls_20261006/report.json')
        if not source['strict_net_source_gain'] or audit['status']!='PASS' or not controls['full_compound_source_control_gate']:
            raise ValueError('source and control gates required')
        mask=ROOT/'artifacts/mendeley_positive_contact_source_20261006/fit_contact_support.png'
        self.pins[str(mask)]=sha256(mask)
        self.support=np.asarray(Image.open(mask))>0
        if self.support.shape!=(50,100) or int(self.support.sum())!=source['support_pixels'] or self.support.sum()<6:
            raise ValueError('source support drift')
        self.threshold=source['fit_threshold']
        repo=Path('E:/PythonProject10/models/dinov2');checkpoint=repo/'weights/dinov2_vits14_pretrain.pth'
        self.pins[str(checkpoint)]=sha256(checkpoint)
        if self.pins[str(checkpoint)]!='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9':raise ValueError('encoder drift')
        self.pins.update({str(p):sha256(p) for p in [repo/'hubconf.py',*sorted((repo/'dinov2').rglob('*.py'))]})
        torch.set_num_threads(8)
        self.encoder=torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=False)
        self.encoder.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);self.encoder.eval()
        self.mean=torch.tensor([.485,.456,.406])[:,None,None];self.std=torch.tensor([.229,.224,.225])[:,None,None]
    def infer(self, patch):
        def unpack(raw):
            m={k:np.asarray(raw[k]) if k in ['center','scale','weights'] else raw[k] for k in ['center','scale','weights','bias']}
            return m,{int(k):v for k,v in raw['calibration'].items()}
        f=descriptor(patch)
        old=agreement({v:predict(*unpack(raw),feature_view(f,v)) for v,raw in self.raw_old.items()})['visual_label_candidate']
        cm,cc=unpack(self.raw_color);sm,sc=unpack(self.raw_semantic)
        color=prediction(cm,color_descriptor(patch),cc)['visual_label_candidate']
        rgb=np.asarray(Image.fromarray(patch).resize((224,112),Image.Resampling.BILINEAR)).copy()
        tensor=(torch.from_numpy(rgb.transpose(2,0,1)).float()/255-self.mean)/self.std
        with torch.inference_mode():tokens=self.encoder.forward_features(tensor.unsqueeze(0))['x_norm_patchtokens'].squeeze(0).cpu().numpy()
        tokens=tokens/np.maximum(np.linalg.norm(tokens,axis=1,keepdims=True),1e-8)
        sf,_=semantic_descriptor(tokens,np.asarray(self.raw_semantic['centers']))
        semantic=prediction(sm,sf,sc)['visual_label_candidate']
        score=float(bright_contacts(patch)[self.support].mean())
        candidate=resolve(old,color,semantic,score>self.threshold)
        return dict(visual_label_candidate=candidate,old_candidate=old,color_candidate=color,
            semantic_candidate=semantic,contact_score=score,positive_contact_cue=score>self.threshold,
            phenotype='mating_body_visible' if candidate==1 else 'socket_contacts_exposed' if candidate==0 else 'uncertain',
            independent_observer_count=1,new_confirmed_connections=0,electrical_continuity='not_assessed')
