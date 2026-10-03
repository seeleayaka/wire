"""Read-only assistant review packet. No operator declarations or assigned edges."""
import sys,json,copy,html
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
sys.path.insert(0,str(REPO))
import cv2,numpy as np
from PIL import Image,ImageDraw
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.terminal_mapping import image_binding,review_mapped_topology
OUT=ROOT/'artifacts/port_endpoint_review_20261002'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Fresh evidence directory required')
    OUT.mkdir();protocol=load(ROOT/'artifacts/sam_crop_coverage_20261001/frozen_protocol.json')
    packet=[];protected={};sections=[]
    for case in protocol['cases']:
        mapping_path=ROOT/'artifacts/terminal_mapping_topology_20261001'/(case['case']+'_draft.json')
        endpoint_path=ROOT/'artifacts/sam_crop_coverage_20261001'/(case['case']+'_source_endpoints.json')
        audit_path=ROOT/'artifacts/topology_duplicates_20261002_v3'/(case['case']+'_audit.json')
        mapping=load(mapping_path);endpoints=load(endpoint_path);audit=load(audit_path)
        source=Path(case['source_binding']['image_path'])
        assert image_binding(source)==mapping['image_binding']==endpoints['image_binding']==case['source_binding']
        for p in (mapping_path,endpoint_path,audit_path,source):protected[str(p)]=sha(p)
        manifest=load(Path(case['fresh_run_directory'])/'run_manifest.json')
        for name,value in {**manifest['verified_files'],**manifest['runtime_fingerprints']}.items():
            assert sha(name)==value;protected[name]=value
        assessment=review_mapped_topology(mapping,source,endpoints)
        assert assessment['decision']=='insufficient_evidence' and assessment['observed_graph'] is None
        save(OUT/(case['case']+'_formal_assessment.json'),assessment)
        records={r['record_id']:r for r in endpoints['records']}
        with Image.open(source) as im:original=im.convert('RGB');pixels=np.asarray(original)
        cards=[]
        for i,(port,diagnostic) in enumerate(zip(mapping['ports'],audit['ports']),1):
            assert port['id']==diagnostic['port_id']
            # Nearest geometric candidates are visual diagnostics ONLY, not an assignment shortlist.
            ids=list(dict.fromkeys(d['record_id'] for d in diagnostic['closest_geometry_diagnostic']))
            masks=[];candidate_rows=[]
            for rid in ids:
                record=records[rid];mask_path=Path(case['fresh_run_directory'])/'sam'/f"mask_{record['source_mask_id']:03d}.png"
                with Image.open(mask_path) as im:binary=np.asarray(im.convert('L'))>0
                _,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),connectivity=8)
                mask=labels==record['component_id'];assert int(mask.sum())==record['component_pixels']
                masks.append(mask)
                candidate_rows.append(dict(record_id=rid,source_score=record['source_score'],
                    meets_existing_comparison_score=record['source_score']>=assessment['minimum_confidence'],
                    geometry_pair_eligible=record['geometry_pair_eligible'],visible_ends_xy=record['visible_ends_xy'],
                    actual_terminal_identity=None,wire_label_reading=None,relationship=None))
            ox,oy,_,_=case['crop_box_xyxy'];bounds=list(port['bbox_xyxy'])
            for rid in ids:
                l,t,r,b=records[rid]['crop_evidence']['component_bbox_in_crop_xyxy']
                bounds=[min(bounds[0],l+ox),min(bounds[1],t+oy),max(bounds[2],r+ox),max(bounds[3],b+oy)]
            l=max(0,bounds[0]-16);t=max(0,bounds[1]-16);r=min(original.width,bounds[2]+16);b=min(original.height,bounds[3]+16)
            context=[l,t,r,b]
            annotated=pixels.copy();colors=[(255,130,30),(30,150,255),(175,70,200)]
            for mask,color in zip(masks,colors):
                region=annotated[oy:oy+mask.shape[0],ox:ox+mask.shape[1]]
                region[mask]=(.55*region[mask]+.45*np.array(color)).astype(np.uint8)
            overlay=Image.fromarray(annotated);draw=ImageDraw.Draw(overlay)
            draw.rectangle(port['bbox_xyxy'],outline=(255,220,0),width=1)
            for rid,color in zip(ids,colors):
                for x,y in records[rid]['visible_ends_xy']:draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
            raw_crop=original.crop(tuple(context));overlay_crop=overlay.crop(tuple(context))
            scale=4;pw=raw_crop.width*scale;ph=raw_crop.height*scale
            card=Image.new('RGB',(pw*2,ph+62),'white');cd=ImageDraw.Draw(card)
            cd.text((8,5),f"{case['case']} draft port {i} | original / model evidence | NOT CONFIRMED",fill='black')
            cd.text((8,22),'  '.join(f"{row['record_id']}={row['source_score']:.3f}" for row in candidate_rows),fill='black')
            cd.text((8,39),f"source crop {context}; display x4; yellow=draft ROI",fill='black')
            card.paste(raw_crop.resize((pw,ph),Image.Resampling.NEAREST),(0,62))
            card.paste(overlay_crop.resize((pw,ph),Image.Resampling.NEAREST),(pw,62))
            filename=f"{case['case']}_port_{i}.png";card.save(OUT/filename)
            row=dict(port_id=port['id'],port_confirmed=False,draft_roi=port['bbox_xyxy'],
                context_source_xyxy=context,source_binding=case['source_binding'],
                candidates=candidate_rows,original_diagnostic=diagnostic,
                review_status='awaiting_operator_identity_and_seating_review',human_conclusion=None,
                assignment=None,expected_connections=None,card=filename)
            cards.append(row)
            sections.append(f'<section><h2>{html.escape(case["case"])} · 入口草稿 {i}</h2><img src="{filename}" alt="原图与模型证据对照"><p>端子身份、线号和插接关系均未确认；颜色仅表示模型证据。</p></section>')
        packet.append(dict(case=case['case'],image_binding=case['source_binding'],ports=cards,
            formal_assessment_file=case['case']+'_formal_assessment.json',confirmed_connections=[],
            no_port_map=not mapping['ports']))
    assert {name:sha(name) for name in protected}==protected
    save(OUT/'review_packet.json',dict(record_kind='assistant_preliminary_evidence_packet',cases=packet,
        operator_review=False,new_inference=False,minimum_comparison_score=.75,protected_files=protected,
        automatic_connections=[],existing_files_unchanged=True))
    page='<!doctype html><meta charset="utf-8"><title>端点归属核查</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;color:#30343b;background:#f7f8fa}section{background:white;padding:20px;margin:20px 0;border:1px solid #ddd}img{width:100%;image-rendering:pixelated}</style><h1>端点归属核查</h1><p>原图局部与现有模型证据。放大仅供观察，没有新增图像细节。此包不是人工确认，也不生成连接边。</p>'+''.join(sections)+'<p>视角2尚无接线入口地图，不能套用视角1的坐标或编号。</p>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print('REVIEW PACKET COMPLETE',flush=True)
if __name__=='__main__':main()
