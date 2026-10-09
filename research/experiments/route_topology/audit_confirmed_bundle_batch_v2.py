"""Independent raw-pixel support/feature/model/decision replay of fresh demo.

SAM file integrity uses the existing verifier; native component support, feature
arithmetic, frozen logits/ranks and typed decision rules are separately replayed.
The local-pose producer is not claimed to be a second independent estimator.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_prompt_contrast import verify
from run_review import save,verified_run

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    source=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_review_20261006'
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    preparation=json.loads((evidence/'preparation_report.json').read_text(encoding='utf-8'))
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    models=json.loads((ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005/model.json').read_text(encoding='utf-8'))
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins'] or any(sha256(p)!=d for p,d in report['pins'].items()):raise ValueError('evidence/code/E drift')
    observed={r['id']:r['observation'] for r in report['cases']};observed['reference']=report['reference_observation']
    replays={};raw_mask_count=0
    for row in preparation['cases']:
        original=row['original_source'];case=row.get('crop_context');expected=observed[row['id']]
        if sha256(original['path'])!=original['image_sha256']:raise ValueError('original SHA drift')
        with Image.open(original['path']) as im:rgb=np.asarray(im.convert('RGB'))
        if case is None:
            if row['sam_inference_requested']:raise ValueError('SAM requested without reliable crop')
            a=b=c=d=0
        else:
            a,b,c,d=case['crop_box_xyxy'];fresh_crop=rgb[b:d,a:c]
            with Image.open(case['source']['path']) as im:stored=np.asarray(im.convert('RGB'))
            if not np.array_equal(fresh_crop,stored):raise ValueError('SAM crop not current original pixels')
        poses={p['id']:p for p in row['anchors']};supported={}
        for identity,p in poses.items():
            supported[identity]=bool(p.get('localization_proposal_supported') is True and p.get('gates')
                and all(type(v) is bool and v for v in p['gates'].values()))
        if 'descriptor' not in row:
            if row['sam_inference_requested'] or row['phenotype']!='uncertain' or supported.get('FAN_CPU',False):
                raise ValueError('missing qualified socket feature cannot yield a definite decision')
            state='uncertain'
        else:
            socket=poses['FAN_CPU'];h=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(socket['inspection_to_reference_local'])
            patch=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
            with Image.open(row['socket_patch_path']) as im:saved=np.asarray(im.convert('RGB'))
            if not np.array_equal(patch,saved) or sha256(row['socket_patch_path'])!=row['socket_patch_sha256']:raise ValueError('fresh socket pixel reconstruction differs')
            lab=cv2.cvtColor(patch,cv2.COLOR_RGB2LAB).astype(float)/255;gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY).astype(float)/255
            dx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)/8;dy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)/8
            magnitude=np.minimum(1,np.sqrt(dx**2+dy**2));angle=np.arctan2(dy,dx)%np.pi;features=[]
            for y in range(4):
                for x in range(8):
                    area=(slice(y*50//4,(y+1)*50//4),slice(x*100//8,(x+1)*100//8))
                    features.extend(np.mean(lab[area],axis=(0,1)));features.extend(np.std(lab[area],axis=(0,1)))
                    for k in range(4):features.append(float(np.mean(magnitude[area]*((angle[area]>=k*np.pi/4)&(angle[area]<(k+1)*np.pi/4)))))
            f=np.asarray(features)
            if not np.allclose(f,row['descriptor'],rtol=1e-12,atol=1e-14):raise ValueError('feature replay differs')
            feature_labels=[]
            for view,m in models.items():
                ff=f.copy();color=np.arange(320)%10<6
                if view=='color':ff[~color]=0
                elif view=='edge':ff[color]=0
                logit=np.dot((ff-m['center'])/m['scale'],m['weights'])+m['bias']
                p=float(1/(1+np.exp(-np.clip(logit,-40,40))))
                ranks={k:(1+sum(v>=candidate for v in m['calibration'][str(k)]))/(len(m['calibration'][str(k)])+1)
                    for k,candidate in [(0,p),(1,1-p)]}
                admitted=[k for k in [0,1] if ranks[k]>.05];label=admitted[0] if len(admitted)==1 else None
                if label!=row['socket_evidence']['feature_view_predictions'][view]['visual_label_candidate']:raise ValueError('feature phenotype differs')
                feature_labels.append(label)
            label=feature_labels[0] if feature_labels[0] is not None and len(set(feature_labels))==1 else None
            state='mating_body_visible' if label==1 else 'socket_contacts_exposed' if label==0 else 'uncertain'
        if state!=row['phenotype']:raise ValueError('socket agreement differs')
        eligible=[];high_socket=False;mask_audit={r['record_id']:r for r in expected['mask_audit']}
        if row['sam_inference_requested']:
            for recipe in protocol['recipes']:
                run=evidence/row['id']/recipe
                manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
                provenance=verified_run(run,manifest['image_binding']['image_path'])
                mask_paths=provenance['paths'];scores=provenance['scores']
                raw_mask_count+=len(mask_paths)
                if recipe!='cable_plus_reference_anatomy_box':continue
                for i,path in enumerate(mask_paths):
                    with Image.open(path) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
                    identity=f'mask_{i+1:03d}';score=scores[i];record=mask_audit[identity]
                    if hashlib.sha256(raw.tobytes()).hexdigest()!=record['mask_array_sha256']:raise ValueError('mask pixel hash differs')
                    yy,xx=np.nonzero(raw);height,width=raw.shape
                    boundary=bool(len(xx) and (xx.min()<=1 or yy.min()<=1 or xx.max()>=width-2 or yy.max()>=height-2))
                    if boundary!=record['boundary_truncated']:raise ValueError('boundary gate differs')
                    count,labels,stats,_=cv2.connectedComponentsWithStats(raw,connectivity=8)
                    if len(record['components'])!=count-1:raise ValueError('native components lost')
                    for part in range(1,count):
                        ys,xs=np.nonzero(labels==part);points=np.array([xs+a,ys+b,np.ones(len(xs))])
                        hits={}
                        for anchor in scope['anchors']:
                            identity_anchor=anchor['id'];hits[identity_anchor]=0
                            if not supported.get(identity_anchor,False):continue
                            q=np.array(poses[identity_anchor]['inspection_to_reference_local'])@points
                            q=q[:2]/q[2:3];left,top,right,bottom=anchor['bbox_xyxy']
                            hits[identity_anchor]=int(np.count_nonzero((q[0]>=left)&(q[0]<=right)&(q[1]>=top)&(q[1]<=bottom)))
                        expected_part=record['components'][part-1]
                        if hits!=expected_part['anchor_pixel_support'] or len(xs)!=expected_part['pixel_count']:raise ValueError('native anchor support independently differs')
                        valid=score>=.75 and not boundary and all(supported.values()) and all(v>0 for v in hits.values())
                        if valid:eligible.append((identity,part))
                        if score>=.75 and hits['FAN_CPU']>0:high_socket=True
        if len(eligible)!=len(expected['eligible_native_components']) or high_socket!=expected['high_score_mask_touches_socket']:
            raise ValueError('component qualification/count differs')
        replays[row['id']]={'socket_state':state,'unique':len(eligible)==1,'supported':supported,
            'high_socket':high_socket,'sam_verified':row['sam_inference_requested']}
    ref=replays['reference'];reference_ready=(scope['reference_review']['confirmed'] is True and ref['sam_verified']
        and ref['socket_state']=='mating_body_visible' and ref['unique'] and all(ref['supported'].values()))
    for result in report['cases']:
        r=replays[result['id']];decision='insufficient_evidence'
        if reference_ready and r['sam_verified'] and r['supported'].get('FAN_CPU',False):
            if r['socket_state']=='socket_contacts_exposed' and not r['high_socket']:decision='visible_socket_attachment_change_supported'
            elif r['socket_state']=='mating_body_visible' and r['unique'] and all(r['supported'].values()):decision='same_visible_bundle_attachment_supported'
        if decision!=result['comparison']['decision']:raise ValueError('independent typed decision differs')
        if result['comparison']['single_wire_connection_confirmed'] or result['comparison']['electrical_disconnections_confirmed']:
            raise ValueError('bundle wrongly promoted to single wire/electrical result')
    if raw_mask_count!=report['SAM_output_instances'] or source_pins()!=protocol['mainline_pins']:raise ValueError('inventory/E drift')
    output=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_audit_20261006';output.mkdir(exist_ok=False)
    audit={'status':'PASS','original_images_reconstructed':len(preparation['cases']),'socket_feature_reconstructions':sum('descriptor' in r for r in preparation['cases']),'native_masks_verified':raw_mask_count,
        'native_component_anchor_support_independently_replayed':True,'typed_decisions_independently_replayed':True,
        'local_pose_is_shared_producer_not_second_independent_estimator':True,
        'reference_review_confirmed':True,'electrical_connections_confirmed':0,'not_field_accuracy':True,
        'pins':{str(p):sha256(p) for p in [Path(__file__),source/'report.json',evidence/'protocol.json',evidence/'preparation_report.json']}}
    save(output/'report.json',audit);print(json.dumps(audit))


if __name__=='__main__':main()
