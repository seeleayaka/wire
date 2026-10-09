"""Single source-only frozen trial; stop without demo inference if no strict gain."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from core import sha256
from component_cues import fit_head, probability, prediction, color_descriptor, gate
from socket_hog_candidate import descriptor, resolve, POLICY
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=ROOT/'artifacts/mendeley_hog_socket_source_v2_20261007'
    folder.mkdir(exist_ok=False)
    lp=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    pp=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json'
    bp=ROOT/'artifacts/mendeley_semantic_visible_pose_rescue_20261006/report.json'
    cp=ROOT/'artifacts/mendeley_component_color_source_20261006/model.json'
    base=json.loads(bp.read_text(encoding='utf-8'))
    assert base['strict_net_source_gain'], 'baseline source gate required'
    partition=json.loads(pp.read_text(encoding='utf-8'))
    labels=json.loads(lp.read_text(encoding='utf-8'))
    by_id={r['id']:r for r in labels['rows']}
    fitting=[by_id[i] for i in partition['fit_ids']]
    calibration=[by_id[i] for i in partition['calibration_ids']]
    previous={c['id']:c for c in base['cases']}
    assert set(previous)==set(partition['calibration_ids'])
    raw_color=json.loads(cp.read_text(encoding='utf-8'))
    color_model={k:np.asarray(raw_color[k]) if k in ['center','scale','weights'] else raw_color[k]
                 for k in ['center','scale','weights','bias']}
    color_cal={int(k):v for k,v in raw_color['calibration'].items()}
    before=source_pins()
    files=[lp,pp,bp,cp,Path(__file__),Path(__file__).with_name('socket_hog_candidate.py'),
           Path(__file__).with_name('component_cues.py')]
    pins={str(p):sha256(p) for p in files}
    save(folder/'protocol.json', {'policy':POLICY,'pins':pins,'mainline_pins':before,
        'fit_ids':partition['fit_ids'],'calibration_ids':partition['calibration_ids'],
        'fit_nominal_only':True,'original_decodes_fresh':True,'historical_qualified_source_poses_reused':True,
        'source_labels_are_prior_assistant_qualitative_not_human_GT':True,
        'reused_development_calibration_not_new_holdout':True,'inspection_inputs_or_labels_read':False,
        'unchanged_head':{'l2':.02,'learning_rate':.05,'iterations':1000,'std_floor':.01},
        'acceptance':'zero wrong, zero old loss, strict positive source gain; all nine shape/color labels1',
        'stop_if':'no source gain or source/code/E drift; no threshold/split/feature sweeps; no demo inference'})
    start=time.monotonic(); nominal={}; probes={}; color_probes={}; prepared=[]
    offsets=[(x,y) for x in [-2,0,2] for y in [-2,0,2]]
    for r in fitting+calibration:
        assert sha256(r['source_path'])==r['source_sha256']
        assert r['pose']['localization_proposal_supported']
        rgb=np.asarray(Image.open(r['source_path']).convert('RGB'))
        pose=np.asarray(r['pose']['inspection_to_reference_local'])
        active_offsets=offsets if r['id'] in previous else [(0,0)]
        features=[]; colors=[]
        for x,y in active_offsets:
            h=np.array([[1,0,-1600+x],[0,1,-1000+y],[0,0,1]])@pose
            assert cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),h,(100,50),flags=cv2.INTER_NEAREST).all()
            patch=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
            f=descriptor(patch); features.append(f)
            if (x,y)==(0,0):
                assert sha256(r['patch_path'])==r['patch_sha256']
                assert np.array_equal(patch,np.asarray(Image.open(r['patch_path']).convert('RGB')))
                nominal[r['id']]=f
            if r['id'] in previous:
                colors.append(prediction(color_model,color_descriptor(patch),color_cal,excluded_id=r['id']))
        probes[r['id']]=features; color_probes[r['id']]=colors
        prepared.append({'id':r['id'],'source_sha256':r['source_sha256'],'feature':nominal[r['id']].tolist(),
                         'probe_features':[f.tolist() for f in features],'color_probes':colors})
        save(folder/'progress.json',{'status':'extracting_source_HOG','completed':len(prepared),'total':197})
    model=fit_head([nominal[r['id']] for r in fitting],[r['visual_label'] for r in fitting])
    cal={k:[{'id':r['id'],'score':probability(model,nominal[r['id']]) if k==0
             else 1-probability(model,nominal[r['id']])} for r in calibration if r['visual_label']==k] for k in [0,1]}
    results=[];gains=[];losses=[]
    for r in calibration:
        old=previous[r['id']]['prediction']['visual_label_candidate']
        ps=[prediction(model,f,cal,excluded_id=r['id']) for f in probes[r['id']]]
        shape_labels=[p['visual_label_candidate'] for p in ps]
        color_labels=[p['visual_label_candidate'] for p in color_probes[r['id']]]
        final=resolve(old,shape_labels,color_labels)
        if old is None and final==r['visual_label']: gains.append(r['id'])
        if old is not None and final!=old: losses.append(r['id'])
        results.append({'id':r['id'],'visual_label':r['visual_label'],'old_candidate':old,
                        'shape_probes':ps,'color_labels':color_labels,
                        'prediction':{'visual_label_candidate':final}})
    score=gate(results);passed=score['passed'] and not losses and bool(gains)
    assert source_pins()==before and all(sha256(p)==v for p,v in pins.items())
    save(folder/'model.json', {**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()},
                              'calibration':cal,'policy':POLICY})
    save(folder/'report.json',{'status':'complete','source_gate':score,'strict_net_source_gain':passed,
        'source_gains':gains,'source_losses':losses,'baseline_gate':gate(base['cases']),
        'prepared':prepared,'cases':results,'seconds':time.monotonic()-start,
        'protocol_sha256':sha256(folder/'protocol.json'),'inspection_inputs_or_labels_read':False,
        'new_topology_hits':0,'deployed':False,'source_development_not_field_accuracy':True})
    save(folder/'progress.json',{'status':'complete','source_gate_passed':passed})
    print(json.dumps({'strict_net_source_gain':passed,'source_gate':score,'gains':gains,'losses':losses,
                      'seconds':time.monotonic()-start}))


if __name__=='__main__': main()
