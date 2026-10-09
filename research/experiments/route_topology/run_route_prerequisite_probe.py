"""Native-mask tracing readiness: explicitly REPLAY, NOT new SAM inference."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from core import sha256
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_route_prerequisites_20261006';out.mkdir(exist_ok=False)
    before=source_pins(); rows=[]
    gate_path=ROOT/'artifacts/mendeley_harness_source_gate_20261006/report.json'
    old=json.loads(gate_path.read_text(encoding='utf-8'))
    inputs={str(gate_path):sha256(gate_path)}
    folder=ROOT/'artifacts/mendeley_harness_source_controls_20261006'
    for case in old['cases']:
        identity=case['id']; counts=[]
        for path in sorted((folder/identity/'wire_harness_plus_reference_anatomy_box/sam').glob('mask_*.png')):
            if path.name=='mask_union.png': continue
            inputs[str(path)]=sha256(path)
            mask=np.asarray(Image.open(path).convert('L'))>0
            sk=skeletonize(mask)
            degree=cv2.filter2D(sk.astype(np.uint8),cv2.CV_16U,np.ones((3,3),np.uint8))-sk
            components=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)[0]-1
            junction_clusters=cv2.connectedComponents((sk & (degree>2)).astype(np.uint8),connectivity=8)[0]-1
            counts.append(dict(mask_path=str(path),native_pixels=int(mask.sum()),skeleton_pixels=int(sk.sum()),
                               components=components,endpoints=int((sk & (degree==1)).sum()),
                               junction_clusters=junction_clusters,added_bridge_pixels=0))
        eligible=case['observation']['eligible_native_components']
        rows.append(dict(id=identity,masks=counts,two_anchor_native_components=len(eligible),
                         original_decision=case['comparison']['decision'],
                         source_annotation=case['source_annotation']))
    repo=Path('E:/PythonProject10/data/external_datasets/external_repos_20260826/rt_dlo')
    checkpoint=repo/'checkpoints/CP_segmentation.pth'
    report=dict(status='complete',cases=rows,pins=inputs,mainline_unchanged=before==source_pins(),
        mBEST_style=dict(status='blocked_by_input_segmentation',
            reason='visible01 has no continuous two-anchor native component; crossing pairing cannot fill occlusion',
            native_mask_replay=True,fresh_SAM_inference=False,full_official_algorithm_executed=False,
            no_morphological_closing=True,new_confirmed_connections=0),
        RT_DLO=dict(status='not_run_missing_assets',repo_exists=repo.exists(),checkpoint_exists=checkpoint.exists(),
            official_source='https://github.com/lar-unibo/RT-DLO',
            historical_report_is_not_current_inference=True),
        known_graph=dict(status='retain_existing_route_invariance',
            limitation='expected graph constrains matching but cannot invent missing observed edges'),
        deployed=False,not_field_accuracy=True)
    if not report['mainline_unchanged']: raise ValueError('mainline drift')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(out/'report.json'),masks=sum(len(r['masks']) for r in rows),
        geometry=[dict(id=r['id'],two_anchor_components=r['two_anchor_native_components'],masks=r['masks']) for r in rows],RT_DLO=report['RT_DLO'])))
if __name__=='__main__': main()
