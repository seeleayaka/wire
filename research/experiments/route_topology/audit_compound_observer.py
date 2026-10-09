"""Independent composition entry replay; saved poses, fresh originals and encoder."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from compound_socket_observer import Observer
from core import sha256
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    output=ROOT/'artifacts/mendeley_compound_observer_audit_20261006';output.mkdir(exist_ok=False)
    before=source_pins();observer=Observer();rows=[];files={}
    for name in ['mendeley_compound_fresh_controls_20261006','mendeley_compound_fresh30_20261006']:
        folder=ROOT/'artifacts'/name
        protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        assert before==protocol['mainline_pins']
        pins=json.loads((folder/'model_pins.json').read_text(encoding='utf-8'))
        assert all(sha256(p)==d for p,d in pins.items())
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        files[str(folder/'report.json')]=sha256(folder/'report.json')
        for row in report['cases']:
            origin=row['source'];assert sha256(origin['path'])==origin['image_sha256']
            rgb=np.asarray(Image.open(origin['path']).convert('RGB'))
            matrix=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.asarray(row['pose']['inspection_to_reference_local'])
            patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
            assert np.array_equal(patch,np.asarray(Image.open(folder/(row['id']+'_fresh_socket.png')).convert('RGB')))
            current=observer.infer(patch)
            assert current['visual_label_candidate']==row['compound_candidate']
            for key in ['old_candidate','color_candidate','semantic_candidate','positive_contact_cue','contact_score']:
                assert current[key]==row[key],(name,row['id'],key)
            rows.append(dict(dataset=name,id=row['id'],prediction=current))
            (output/'progress.json').write_text(json.dumps(dict(status='running',completed=len(rows),total=36)),encoding='utf-8')
    assert before==source_pins() and all(sha256(p)==d for p,d in {**observer.pins,**files}.items())
    result=dict(status='PASS',cases=rows,original_decodes_and_encoder_fresh=True,pose_matrices_reused_explicitly=True,
        descriptor_functions_reused_not_fully_independent_algorithms=True,mainline_pins=before,pins=observer.pins,
        electrical_or_topology_acceptance=False,mainline_unchanged=True)
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'progress.json').write_text(json.dumps(dict(status='complete',completed=len(rows))),encoding='utf-8')
    print(json.dumps(dict(status='PASS',cases=len(rows))))
if __name__=='__main__':main()
