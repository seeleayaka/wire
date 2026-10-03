"""Reconstruct bound existing components. No SAM rerun, tuning or source edits."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
WORK=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
sys.path.insert(0,str(REPO))
import cv2,numpy as np
from PIL import Image,ImageDraw
from inspection_agent.terminal_mapping import image_binding
from inspection_agent.optional_port_crop_review import sha
from visible_evidence_duplicates import audit_duplicates,endpoint_port_diagnostics
BASE=WORK/'artifacts/sam_crop_coverage_20261001';OUT=WORK/'artifacts/topology_duplicates_20261002_v3'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Do not overwrite evidence')
    OUT.mkdir();protocol=load(BASE/'frozen_protocol.json');protected={};summaries=[]
    save(OUT/'protocol.json',dict(mask_iou_review_threshold=.8,exact_only_deduplication=True,
        inference=False,no_nearest_assignment=True,no_threshold_tuning=True,cases=[c['case'] for c in protocol['cases']]))
    for case in protocol['cases']:
        source=Path(case['source_binding']['image_path']);assert image_binding(source)==case['source_binding']
        assert image_binding(Path(case['crop_binding']['image_path']))==case['crop_binding']
        with Image.open(source) as image, Image.open(case['crop_binding']['image_path']) as crop:
            assert np.array_equal(np.asarray(image.convert('RGB').crop(tuple(case['crop_box_xyxy']))),np.asarray(crop.convert('RGB')))
        run=Path(case['fresh_run_directory']);manifest=load(run/'run_manifest.json')
        for name,fingerprint in {**manifest['verified_files'],**manifest['runtime_fingerprints']}.items():
            assert sha(name)==fingerprint;protected[name]=fingerprint
        path=BASE/(case['case']+'_source_endpoints.json');report=load(path);protected[str(path)]=sha(path)
        assert report['image_binding']==case['source_binding']
        assert report['coordinate_transform']['box_xyxy']==case['crop_box_xyxy']
        masks=[];records=report['records'];sam=load(run/'sam/report.json')
        decoded={}
        for record in records:
            maskid=record['source_mask_id'];mp=run/'sam'/f'mask_{maskid:03d}.png'
            if maskid not in decoded:
                with Image.open(mp) as image:binary=np.asarray(image.convert('L'))>0
                assert list(binary.shape[::-1])==case['crop_binding']['image_size']
                decoded[maskid]=cv2.connectedComponentsWithStats(binary.astype(np.uint8),connectivity=8)
            _,labels,stats,_=decoded[maskid];component=labels==record['component_id']
            assert int(component.sum())==record['component_pixels']
            assert record['source_score']==sam['scores'][maskid-1]
            cid=record['component_id'];x,y,w,h=[int(stats[cid,k]) for k in (cv2.CC_STAT_LEFT,cv2.CC_STAT_TOP,cv2.CC_STAT_WIDTH,cv2.CC_STAT_HEIGHT)]
            assert record['crop_evidence']['component_bbox_in_crop_xyxy']==[x,y,x+w,y+h]
            masks.append(component)
        audit=audit_duplicates(masks,records)
        mapping_path=WORK/'artifacts/terminal_mapping_topology_20261001'/(case['case']+'_draft.json')
        mapping=load(mapping_path);assert mapping['image_binding']==case['source_binding']
        protected[str(mapping_path)]=sha(mapping_path)
        audit.update(case=case['case'],ports=endpoint_port_diagnostics(records,mapping['ports']),
            expected_connections_known=mapping['expected_connections'] is not None,
            topology_decision='insufficient_evidence',observed_graph=None)
        save(OUT/(case['case']+'_audit.json'),audit)
        # Review first six strongest ambiguous pairs; never auto-merge them.
        chosen=sorted(audit['high_overlap_pairs']+audit['contained_overlap_pairs'],key=lambda p:-p['mask_iou'])[:6]
        index={r['record_id']:i for i,r in enumerate(records)}
        with Image.open(source) as im:original=np.asarray(im.convert('RGB'))
        panels=[];ox,oy,_,_=case['crop_box_xyxy']
        for pair in chosen:
            a,b=[masks[index[r]] for r in pair['record_ids']];ys,xs=np.where(a|b)
            x1=max(0,int(xs.min())+ox-8);x2=min(original.shape[1],int(xs.max())+ox+9)
            y1=max(0,int(ys.min())+oy-8);y2=min(original.shape[0],int(ys.max())+oy+9)
            colors=original.copy()
            region=colors[oy:oy+a.shape[0],ox:ox+a.shape[1]]
            region[a&~b]=(255,130,30);region[b&~a]=(30,150,255);region[a&b]=(30,220,130)
            panel=Image.fromarray(colors[y1:y2,x1:x2]).resize((300,240))
            tile=Image.new('RGB',(300,275),'white');tile.paste(panel,(0,35))
            ImageDraw.Draw(tile).text((5,5),' / '.join(pair['record_ids'])+f" IoU {pair['mask_iou']:.3f}",fill='black')
            panels.append(tile)
        if panels:
            sheet=Image.new('RGB',(900,275*((len(panels)+2)//3)),'white')
            for i,panel in enumerate(panels):sheet.paste(panel,((i%3)*300,(i//3)*275))
            sheet.save(OUT/(case['case']+'_overlap_review.png'))
        summaries.append({k:audit[k] for k in ('case','component_records','exact_mask_groups','exact_duplicate_redundancy',
            'eligible_records','eligible_exact_mask_groups','ports','expected_connections_known','topology_decision')})
        summaries[-1]['high_overlap_pairs']=len(audit['high_overlap_pairs'])
        summaries[-1]['contained_overlap_pairs']=len(audit['contained_overlap_pairs'])
        summaries[-1]['maximum_pair_mask_iou']=audit['maximum_pair_mask_iou']
    assert {name:sha(name) for name in protected}==protected
    save(OUT/'report.json',dict(cases=summaries,existing_evidence_unchanged=True,new_inference=False,
        automatic_connections=[],physical_wire_counts_claimed=False))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
