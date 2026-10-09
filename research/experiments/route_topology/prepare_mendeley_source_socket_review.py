"""Source-only slot occupancy annotation intake; never infer state from filename.

All40 source rerouting and40 source disconnection photos are shown for actual
scope review. Class metadata selects the review pool, not occupancy labels.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from core import sha256,image_binding
from heldout_anchor_pose import localize
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True


def main():
    source=ROOT/'artifacts/mendeley_visible_fan_scope_20261005'
    old=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'));anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    ref_path=Path(old['original_sources']['reference']['path']);folder=ref_path.parent
    pools={'source_routing':sorted(folder.glob('misrouted_*.JPG')),
           'source_disconnection':sorted(folder.glob('disconnected_*.JPG'))}
    if [len(pools[k]) for k in pools]!=[40,40]:raise ValueError('frozen all80 source review inputs required')
    output=ROOT/'artifacts/mendeley_source_socket_occupancy_review_20261005'
    if output.exists():raise FileExistsError('preserve prior source review')
    files=[Path(__file__),Path(__file__).with_name('heldout_anchor_pose.py'),Path(__file__).with_name('local_anchor_pose.py'),
        Path(__file__).with_name('prepare_mendeley_scope.py'),scope_path,ref_path,
        *[p for group in pools.values() for p in group],Path('E:/PythonProject10/prototype/assembly_auto_review_robust_v3.py'),
        Path('E:/PythonProject10/prototype/assembly_auto_review_robust_v2.py'),Path('E:/PythonProject10/prototype/assembly_auto_review_dino.py')]
    pins={str(p):sha256(p) for p in files};mainline=source_pins();output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),'pins':pins,'mainline_pins':mainline,
        'pools':{k:[str(p) for p in group] for k,group in pools.items()},
        'source_class_metadata_only_selects_pool_not_occupancy_truth':True,
        'inspection_images_or_labels_read':False,'automatic_occupancy_labels_created':0,
        'reference_review_confirmed':False,'stop_if':'source/code/E drift or failed source localization;15min between-case'})
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    with Image.open(ref_path) as opened:reference=np.asarray(opened.convert('RGB'))
    begun=time.perf_counter();rows=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    try:
        for pool,paths in pools.items():
            page=None
            for index,path in enumerate(paths):
                if time.perf_counter()-begun>900:raise TimeoutError('15min between-case limit')
                identity=f'{pool}_{index+1:03d}'
                save(output/'progress.json',{'status':'fresh_source_preparation','current_id':identity,
                    'completed':len(rows),'total':80,'seconds':time.perf_counter()-begun})
                if sha256(path)!=pins[str(path)]:raise ValueError('original source changed')
                with Image.open(path) as opened:inspection=np.asarray(opened.convert('RGB'))
                _,registration=automatic_homography(reference[:,:,::-1].copy(),inspection[:,:,::-1].copy())
                if not registration['alignment_quality']['reliable']:raise ValueError('source global registration failed')
                pose=localize(reference,inspection,anchor,np.asarray(registration['source_to_reference_homography']))
                if not pose['localization_proposal_supported']:raise ValueError('source FAN_CPU localization failed')
                matrix=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.asarray(pose['inspection_to_reference_local'])
                valid=cv2.warpPerspective(np.ones(inspection.shape[:2],np.uint8),matrix,(100,50),flags=cv2.INTER_NEAREST)
                if not np.all(valid):raise ValueError('source socket patch incomplete')
                patch=cv2.warpPerspective(inspection,matrix,(100,50),flags=cv2.INTER_LINEAR)
                patch_path=output/(identity+'_socket.png');Image.fromarray(patch).save(patch_path)
                row={'id':identity,'source_path':str(path),'source_sha256':pins[str(path)],
                    'source_pool':pool,'patch_path':str(patch_path),'patch_sha256':sha256(patch_path),
                    'registration':registration,'pose':pose,'occupancy_label':None,
                    'reviewer_type':None,'reviewer':None,'evidence_note':None,
                    'physical_electrical_continuity_assessed':False}
                rows.append(row)
                if index%20==0:page=Image.new('RGB',(1800,840),'#f4f5f7')
                cell=Image.new('RGB',(360,210),'white');draw=ImageDraw.Draw(cell)
                draw.text((7,4),identity,font=font,fill='#25364a')
                draw.text((7,25),'来源图；占用状态尚未标记',font=font,fill='#926113')
                cell.paste(Image.fromarray(patch).resize((340,170),Image.Resampling.NEAREST),(10,40))
                page.paste(cell,((index%5)*360,((index%20)//5)*210))
                if index%20==19:page.save(output/(pool+f'_page_{index//20+1}.png'))
        if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=mainline:raise ValueError('source/code/E changed')
        save(output/'review_inventory.json',{'status':'prepared','rows':rows,'occupancy_labels_confirmed':0,
            'inspection_images_or_labels_read':False,'automatic_occupancy_labels_created':0,
            'reference_review_confirmed':False,'protocol_sha256':sha256(output/'protocol.json')})
        save(output/'progress.json',{'status':'prepared_for_visual_review','completed':80,'seconds':time.perf_counter()-begun,
            'occupancy_labels_confirmed':0})
        print(json.dumps({'status':'prepared_for_visual_review','sources':80,'seconds':time.perf_counter()-begun,'labels':0}))
    except BaseException as error:
        save(output/'progress.json',{'status':'failed','completed':len(rows),'reason':str(error)})
        (output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
