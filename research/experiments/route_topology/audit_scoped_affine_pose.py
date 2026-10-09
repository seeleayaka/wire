"""Fresh static correspondences, independent heldout residual and dispatch audit.

OpenCV SIFT/BF extraction shared; recorded matrices NOT refitted in this audit.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_prompt_contrast import save,verify
from run_audit import source_pins
ROOT=Path(__file__).resolve().parents[2]

def correspondences(reference,image,anchor,pose,global_matrix):
    box=pose['reference_context_xyxy'];ib=pose['inspection_context_xyxy']
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.014,edgeThreshold=12)
    kr,dr=sift.detectAndCompute(cv2.cvtColor(reference[box[1]:box[3],box[0]:box[2]],cv2.COLOR_RGB2GRAY),None)
    ki,di=sift.detectAndCompute(cv2.cvtColor(image[ib[1]:ib[3],ib[0]:ib[2]],cv2.COLOR_RGB2GRAY),None)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def matches(a,b):
        result={}
        for pair in matcher.knnMatch(a,b,k=2):
            if len(pair)==2 and pair[0].distance/pair[1].distance<.70:result[pair[0].queryIdx]=pair[0].trainIdx
        return result
    forward=matches(di,dr);reverse=matches(dr,di);src=[];dst=[];l,t,r,b=anchor['bbox_xyxy']
    for i,j in sorted(forward.items()):
        if reverse.get(j)!=i:continue
        rp=np.asarray(kr[j].pt)+box[:2];ip=np.asarray(ki[i].pt)+ib[:2]
        mapped=np.asarray(global_matrix)@np.array([*ip,1]);mapped=mapped[:2]/mapped[2]
        if l<=rp[0]<=r and t<=rp[1]<=b:continue
        if l<=mapped[0]<=r and t<=mapped[1]<=b:continue
        src.append(ip);dst.append(rp)
    return np.asarray(src,np.float32),np.asarray(dst,np.float32)

def main():
    source=ROOT/'artifacts/mendeley_scoped_affine_controls_20261006'
    rp=source/'report.json';report=json.loads(rp.read_text(encoding='utf-8'))
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    verify(protocol['pins']);assert report['source_controls_passed'] and report['all_demo_executed']
    sp=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    scope=json.loads(sp.read_text(encoding='utf-8'));reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    before=source_pins();checked=[];dispatches=0;new=[]
    for case in report['cases']:
        assert sha256(case['source']['path'])==case['source']['image_sha256']
        image=None
        for row in case['anchors']:
            old=row['old'];affine=row['affine'];chosen=row['chosen'];dispatches+=1
            expected=old
            if not old['localization_proposal_supported'] and row['kind']=='visible_lead_emergence' and affine['localization_proposal_supported'] and affine.get('gates') and all(v is True for v in affine['gates'].values()):expected=affine
            assert chosen==expected
            rescued=not old['localization_proposal_supported'] and chosen['localization_proposal_supported'];assert rescued==row['rescued']
            if rescued and case['dataset']=='demo30':new.append(dict(case=case['id'],anchor=row['id']))
            if row['kind']!='visible_lead_emergence' or not (case['dataset']=='source_controls' or rescued):continue
            if image is None:image=np.asarray(Image.open(case['source']['path']).convert('RGB'))
            anchor=next(a for a in scope['anchors'] if a['id']==row['id'])
            src,dst=correspondences(reference,image,anchor,affine,case['global_pose_fresh'])
            assert len(src)==affine['mutual_matches']
            indices=[j for j,(x,y) in enumerate(dst) if int.from_bytes(hashlib.sha256(f'{round(float(x))},{round(float(y))}'.encode('ascii')).digest()[:4],'big')%3==0]
            assert len(indices)==affine['heldout_count'] and len(src)-len(indices)==affine['training_count']
            matrix=np.asarray(affine['inspection_to_reference_local']);np.testing.assert_array_equal(matrix[2],[0,0,1])
            errors=[]
            for j in indices:
                mapped=matrix@np.array([*src[j],1]);errors.append(float(np.linalg.norm(mapped[:2]/mapped[2]-dst[j])))
            median=float(np.median(errors));ratio=sum(e<=4 for e in errors)/len(errors)
            assert abs(median-affine['heldout_median_error_px'])<1e-8 and abs(ratio-affine['heldout_support_ratio'])<1e-12
            assert len(indices)>=8 and median<=2.5 and ratio>=.5
            checked.append(dict(case=case['id'],anchor=row['id'],heldout_median_px=median,heldout_points=len(indices)))
    assert dispatches==70 and new==report['newly_supported']
    assert source_pins()==before;verify(protocol['pins'])
    out=ROOT/'artifacts/mendeley_scoped_affine_pose_audit_20261006';out.mkdir(exist_ok=False)
    save(out/'report.json',dict(status='PASS',source_report_sha256=sha256(rp),dispatches_checked=dispatches,
        fresh_heldout_residual_checks=checked,newly_supported=new,
        shared_SIFT_BF_extractor=True,independent_hash_split_residuals_and_dispatch=True,
        does_not_independently_refit_or_verify_identity=True,production_unchanged=True,deployed=False))
    print(json.dumps(dict(status='PASS',newly_supported=new)))
if __name__=='__main__':main()
