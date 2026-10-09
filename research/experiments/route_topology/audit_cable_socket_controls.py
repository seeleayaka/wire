"""One-shot independent replay after fixed source worker; no SAM/model restart."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from audit_semantic_visible_bundle_native import flood_components
from run_prompt_contrast import save,verify
from run_audit import source_pins
from run_paired_evidence import load_run
from analyze_mendeley_scope import render_all

ROOT=Path(__file__).resolve().parents[2]

def replay_hits(parts, width, crop_origin, anchors, poses):
    """Scalar homogeneous mapping of every native pixel, no component pruning."""
    signatures=[]
    for part in parts:
        hits={a['id']:0 for a in anchors}
        for anchor in anchors:
            pose=next(p for p in poses if p['id']==anchor['id'])
            if not pose['localization_proposal_supported'] or not all(pose['gates'].values()):continue
            matrix=np.asarray(pose['inspection_to_reference_local'],dtype=float)
            l,t,r,b=anchor['bbox_xyxy']
            for flat in part:
                y,x=divmod(flat,width);x+=crop_origin[0];y+=crop_origin[1]
                qx=matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]
                qy=matrix[1,0]*x+matrix[1,1]*y+matrix[1,2]
                qw=matrix[2,0]*x+matrix[2,1]*y+matrix[2,2]
                if abs(qw)<1e-9:raise ValueError('native mapping horizon')
                if l<=qx/qw<=r and t<=qy/qw<=b:hits[anchor['id']]+=1
        signatures.append((len(part),hits))
    return signatures

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    evidence=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    out=ROOT/'artifacts/mendeley_cable_socket_controls_audit_20261006'
    out.mkdir(exist_ok=False)
    save(out/'progress.json',dict(status='waiting_for_source_report',no_inference=True))
    started=time.monotonic();digest=sha256(__file__)
    try:
        while True:
            progress=json.loads((evidence/'pipeline_progress.json').read_text(encoding='utf-8'))
            if progress['status']=='complete':break
            if not args.wait:raise FileNotFoundError('source report not complete')
            if progress['status']=='failed':raise RuntimeError('source pipeline failed; no restart')
            if time.monotonic()-started>2400:raise TimeoutError('bounded one-shot wait elapsed')
            time.sleep(20)
        report_path=evidence/'report.json';report=json.loads(report_path.read_text(encoding='utf-8'))
        protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
        prepared=json.loads((evidence/'preparation_report.json').read_text(encoding='utf-8'))['cases']
        contract=json.loads((evidence/'experiment_contract.json').read_text(encoding='utf-8'))
        scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
        assert report['status']=='complete' and len(report['cases'])==len(prepared)==5
        assert {c['id'] for c in prepared}==set(contract['expected_visible_ids']+contract['expected_exposed_ids'])
        verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
        checks=[];views={r['id']:r['observation'] for r in report['cases']}
        socket=next(a['id'] for a in scope['anchors'] if a['kind']=='wire_entry_socket')
        for case in prepared:
            origin=case['original_source'];context=case['crop_context'];crop=context['crop_box_xyxy']
            view=views[case['id']];assert sha256(origin['path'])==origin['image_sha256']==view['source_binding']['image_sha256']
            actual=np.asarray(Image.open(origin['path']).convert('RGB').crop(crop))
            assert np.array_equal(actual,np.asarray(Image.open(context['source']['path']).convert('RGB')))
            recorded={m['record_id']:m for m in view['mask_audit']};eligible=0;high=False;hits=[];count=0
            for recipe in protocol['recipes']:
                run=evidence/case['id']/recipe
                manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
                for filename,d in manifest['verified_files'].items():assert sha256(filename)==d
                source,records=load_run(run);render_all(source,records,out/(case['id']+'_'+recipe))
                if not recipe.endswith('_box'):continue
                scores=json.loads((run/'sam/report.json').read_text(encoding='utf-8'))['scores']
                assert len(scores)==len(recorded)
                for index,score in enumerate(scores,1):
                    identity=f'mask_{index:03d}';old=recorded[identity]
                    raw=np.asarray(Image.open(run/'sam'/(identity+'.png')).convert('L'))>0
                    assert raw.shape==(crop[3]-crop[1],crop[2]-crop[0])
                    parts=flood_components(raw);count+=len(parts)
                    ys,xs=np.where(raw);h,w=raw.shape
                    boundary=bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=w-2 or ys.max()>=h-2))
                    assert boundary==old['boundary_truncated'] and score==old['score']
                    signatures=replay_hits(parts,w,crop[:2],scope['anchors'],case['anchors'])
                    assert sorted((n,tuple(sorted(p.items()))) for n,p in signatures)==sorted((p['pixel_count'],tuple(sorted(p['anchor_pixel_support'].items()))) for p in old['components'])
                    for n,p in signatures:
                        if score>=.75 and not boundary and all(v>0 for v in p.values()):eligible+=1
                        if score>=.75 and p[socket]>0:
                            high=True;hits.append(dict(mask=identity,component_pixels=n,socket_pixels=p[socket]))
            assert eligible==len(view['eligible_native_components']) and high==view['high_score_mask_touches_socket']
            checks.append(dict(id=case['id'],native_components=count,two_anchor_components=eligible,
                high_socket_touch=high,socket_touch_components=hits,appearance=view['socket_state']))
        counterexamples=[r['id'] for r in checks if r['id'] in contract['expected_exposed_ids'] and r['high_socket_touch']]
        assert counterexamples==report['exposed_controls_with_high_socket_touch']
        verify(protocol['pins']);assert source_pins()==protocol['mainline_pins'] and sha256(__file__)==digest
        save(out/'report.json',dict(status='PASS',source_report_sha256=sha256(report_path),cases=checks,
            independently_replayed_native_components_and_scalar_anchor_hits=True,
            exposed_controls_with_high_socket_touch=counterexamples,all_original_pixels_preserved=True,
            source_appearance_evidence_shared_not_independently_retrained=True,
            actual_visual_review='pending',no_decision_override=True,electrical_connections_confirmed=0,deployed=False))
        save(out/'progress.json',dict(status='complete',actual_visual_review='pending',deployed=False))
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=str(error),no_retry=True));raise

if __name__=='__main__':main()
