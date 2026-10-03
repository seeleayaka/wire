"""Audit three pre-existing train01 SAM masks; no inference or detector changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image,ImageDraw
ROOT=Path('E:/PythonProject10');sys.path.insert(0,str(ROOT))


def read(path,flags=cv2.IMREAD_COLOR):
    image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),flags)
    if image is None:raise ValueError('cannot read '+str(path))
    return image


def shape_records(mask,min_pixels=50):
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    records=[]
    for index in range(1,count):
        x,y,w,h,pixels=map(int,stats[index])
        if pixels<min_pixels:continue
        component=labels[y:y+h,x:x+w]==index
        skeleton=skeletonize(component).astype(np.uint8)
        neighbors=cv2.filter2D(skeleton,cv2.CV_16S,np.ones((3,3),np.int16),borderType=cv2.BORDER_CONSTANT)-skeleton
        endpoints=np.argwhere((skeleton>0)&(neighbors==1))
        branch_mask=((skeleton>0)&(neighbors>=3)).astype(np.uint8)
        branch_clusters=cv2.connectedComponents(branch_mask,8)[0]-1
        records.append({'bounds':[x,y,x+w,y+h],'pixels':pixels,'skeleton_pixels':int(skeleton.sum()),
                'end_count':len(endpoints),'branch_clusters':int(branch_clusters),
                'visible_ends':[[int(v[1])+x,int(v[0])+y] for v in endpoints],
                'unambiguous_open_segment':len(endpoints)==2 and branch_clusters==0})
    return records


def load_labels(path,width,height):
    boxes=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        _,cx,cy,w,h=map(float,line.split())
        boxes.append([round((cx-w/2)*width),round((cy-h/2)*height),round((cx+w/2)*width),round((cy+h/2)*height)])
    return boxes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('use fresh output')
    args.output.mkdir(parents=True)
    sources=[ROOT/'manual_review/sam3_mendeley_intelliman_normal_003_thr040_20260828',
             ROOT/'output/mendeley_intelliman_topology_evidence_20260828/sam3_damaged_001_thr040',
             ROOT/'output/mendeley_intelliman_topology_evidence_20260828/sam3_disconnected_001_thr040']
    rows=[];board=Image.new('RGB',(1500,1300),'#20252a')
    for index,directory in enumerate(sources):
        report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
        assert report['prompt']=='cable' and report['confidence_threshold']==.4
        source=Path(report['input']);assert source.parent.name=='train01'
        image=read(source);union=read(directory/'mask_union.png',cv2.IMREAD_GRAYSCALE)>0
        assert image.shape[:2]==union.shape
        masks=[read(directory/f'mask_{n:03d}.png',cv2.IMREAD_GRAYSCALE)>0 for n in range(1,report['instance_count']+1)]
        assert [int(m.sum()) for m in masks]==report['mask_pixel_counts']
        reconstructed=np.logical_or.reduce(masks);assert np.array_equal(reconstructed,union)
        records=[]
        for mask_id,mask in enumerate(masks,1):
            for record in shape_records(mask):records.append({'mask_id':mask_id,**record})
        # Source rectangles are scoring-only fragmented annotations, not cable masks.
        label=source.parents[2]/'labels/train01'/(source.stem+'.txt')
        boxes=load_labels(label,image.shape[1],image.shape[0]) if label.exists() else []
        ratios=[]
        for x0,y0,x1,y1 in boxes:
            crop=union[max(0,y0):min(union.shape[0],y1),max(0,x0):min(union.shape[1],x1)]
            ratios.append(float(crop.mean()) if crop.size else 0.)
        row={'image':source.name,'split':'train01','cache_dir':str(directory),'instances':len(masks),
             'mask_union_fraction':float(union.mean()),'component_count':len(records),
             'open_unbranched_components':sum(r['unambiguous_open_segment'] for r in records),
             'branched_components':sum(r['branch_clusters']>0 for r in records),
             'more_than_two_ends_components':sum(r['end_count']>2 for r in records),
             'components':records,'source_fragment_count':len(boxes),
             'source_fragments_any_mask_pixels':sum(r>0 for r in ratios),
             'source_fragment_mask_fraction':ratios,'report_sha256':hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
             'source_sha256_current':hashlib.sha256(source.read_bytes()).hexdigest(),
             'mask_sha256':hashlib.sha256((directory/'mask_union.png').read_bytes()).hexdigest(),
             'historical_cache_unfingerprinted_source':True}
        rows.append(row)
        panel=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(panel)
        for b in boxes:draw.rectangle(b,outline='red',width=3)
        color=np.asarray(panel).copy();color[union]=(color[union].astype(float)*.55+np.array([0,220,210])*.45).astype(np.uint8)
        panel=Image.fromarray(color);draw=ImageDraw.Draw(panel)
        for record in records:
            for endpoint in record['visible_ends']:draw.ellipse([endpoint[0]-8,endpoint[1]-8,endpoint[0]+8,endpoint[1]+8],fill='yellow')
        panel.thumbnail((1480,390));cell=Image.new('RGB',(1500,433),'#20252a');cell.paste(panel,(0,43))
        label_draw=ImageDraw.Draw(cell);label_draw.text((10,5),source.name+' | masks='+str(len(masks))+' components='+str(len(records))+' branched='+str(row['branched_components']),fill='white')
        label_draw.text((10,23),'Cyan: visible cable masks | Yellow: all skeleton endpoints | Red: source fragments, scoring only',fill='white')
        cell.save(args.output/(source.stem+'.png'));board.paste(cell,(0,index*433))
    board.save(args.output/'cache_audit.png')
    result={'split':'train01','samples':rows,'formal_path_changed':False,
            'warning':'Three historical train caches only; incomplete source fragments are not wire segmentation truth. No inference, no test01 tuning, no new field accuracy or paired fault verdict. Historic source/checkpoint hashes were absent; current hashes cannot prove historic identity.'}
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if k not in ('components','source_fragment_mask_fraction')} for row in rows],indent=2))


if __name__=='__main__':main()
