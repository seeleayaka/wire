"""Reconstruct original inspection pixels and independently replay phenotype."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_phenotype_fresh30_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    model_path=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005/model.json'
    models=json.loads(model_path.read_text(encoding='utf-8'))
    if report['protocol_sha256']!=sha256(source/'protocol.json') or any(sha256(p)!=d for p,d in protocol['pins'].items()):raise ValueError('pins drift')
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    rows=[]
    for row in report['cases']:
        if sha256(row['input_path'])!=row['input_sha256']:raise ValueError('original changed')
        socket=next(a for a in row['anchors'] if a['id']=='FAN_CPU')
        if not socket['localization_proposal_supported']:raise ValueError('unexpected missing supported socket')
        if not all(socket['gates'].values()):raise ValueError('failed spatial gate used')
        with Image.open(row['input_path']) as im:rgb=np.asarray(im.convert('RGB'))
        h=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(socket['inspection_to_reference_local'])
        patch=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
        with Image.open(row['patch_path']) as im:saved=np.asarray(im.convert('RGB'))
        if not np.array_equal(patch,saved) or sha256(row['patch_path'])!=row['patch_sha256']:raise ValueError('fresh pixel replay differs')
        # Independently rebuild the320-dimensional descriptor (not producer import).
        lab=cv2.cvtColor(patch,cv2.COLOR_RGB2LAB).astype(float)/255
        gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY).astype(float)/255
        dx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)/8;dy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)/8
        mag=np.minimum(1,np.sqrt(dx*dx+dy*dy));angle=np.mod(np.arctan2(dy,dx),np.pi);features=[]
        for y in range(4):
            for x in range(8):
                region=(slice(y*50//4,(y+1)*50//4),slice(x*100//8,(x+1)*100//8))
                features.extend(lab[region].mean((0,1)));features.extend(lab[region].std((0,1)))
                for k in range(4):features.append(np.mean(mag[region]*((angle[region]>=k*np.pi/4)&(angle[region]<(k+1)*np.pi/4))))
        f=np.array(features)
        if not np.allclose(f,row['descriptor'],rtol=1e-12,atol=1e-14):raise ValueError('independent features differ')
        predictions=[]
        for view,model in models.items():
            ff=f.copy();mask=np.arange(320)%10<6
            if view=='color':ff[~mask]=0
            if view=='edge':ff[mask]=0
            logit=np.dot((ff-np.array(model['center']))/np.array(model['scale']),model['weights'])+model['bias']
            p=float(1/(1+np.exp(-np.clip(logit,-40,40))))
            rank={k:(1+sum(v>=s for v in model['calibration'][str(k)]))/(1+len(model['calibration'][str(k)])) for k,s in [(0,p),(1,1-p)]}
            admitted=[k for k in [0,1] if rank[k]>.05];predicted=admitted[0] if len(admitted)==1 else None
            expected=row['socket_evidence']['feature_view_predictions'][view]
            if predicted!=expected['visual_label_candidate']:raise ValueError('feature view prediction differs')
            predictions.append(predicted)
        label=predictions[0] if predictions[0] is not None and len(set(predictions))==1 else None
        if label!=row['socket_evidence']['visual_label_candidate']:raise ValueError('agreement differs')
        if row['decision']!='insufficient_evidence' or row['new_confirmed_connections'] or row['confirmed_disconnections']:
            raise ValueError('appearance promoted to connection verdict')
        rows.append({'id':row['id'],'phenotype':row['phenotype'],'socket_original_pixel_replay':True})
    if len(rows)!=30 or source_pins()!=protocol['mainline_pins']:raise ValueError('inventory/E drift')
    output=ROOT/'artifacts/mendeley_socket_phenotype_fresh30_audit_20261005';output.mkdir(exist_ok=False)
    result={'status':'PASS','original_pixel_reconstructions':30,'independent_descriptor_and_predictions_replayed':True,
        'rows':rows,'new_confirmed_connections':0,'confirmed_disconnections':0,'not_field_accuracy':True,
        'pins':{str(p):sha256(p) for p in [Path(__file__),source/'protocol.json',source/'report.json',model_path]}}
    save(output/'report.json',result);print(json.dumps({k:result[k] for k in ['status','original_pixel_reconstructions','new_confirmed_connections']}))


if __name__=='__main__':main()
