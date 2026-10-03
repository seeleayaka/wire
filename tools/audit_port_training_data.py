"""Audit train01 only; no inference, downloads, validation-source or test reads."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from PIL import Image,ImageDraw
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    sys.path.insert(0,str(args.repo))
    from inspection_agent.port_training_audit import parse_boxes,rectangle_labels,inner_split,plan_tiles
    data=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    derived=args.repo/'data/derived/mendeley_port_state_rectseg_20260825'
    image_paths=sorted((data/'images/train01').glob('*.JPG'))
    assert len(image_paths)==240,'unexpected train population'
    split=inner_split([x.name for x in image_paths]);samples=[];hash_groups=defaultdict(list)
    args.output.mkdir(parents=True);start=time.perf_counter()
    selected=[];seen_kinds=set()
    for i,path in enumerate(image_paths,1):
        source_label=data/'labels/train01'/(path.stem+'.txt')
        derived_label=derived/'labels/train'/(path.stem+'.txt')
        with Image.open(path) as image:
            width,height=image.size;exif=image.getexif()
            metadata={str(k):str(exif.get(k,'')) for k in (271,272,306,36867,42033)}
        boxes=parse_boxes(source_label.read_text(encoding='utf-8'),width,height)
        expected=rectangle_labels(boxes)
        actual=derived_label.read_text(encoding='utf-8').splitlines()
        assert expected==actual,'derived label mismatch: '+path.name
        image_sha=sha(path);hash_groups[image_sha].append(path.name)
        tiles=plan_tiles(boxes,width,height);ports=[b for b in boxes if b['source_class'] in (3,4)]
        covered={b['source_box_index'] for tile in tiles if tile['usable'] for b in tile['labels']}
        missing=[index for index,b in enumerate(boxes) if b['source_class'] in (3,4) and index not in covered]
        samples.append({'image':path.name,'kind':path.stem.split('_')[0],'source_split':'train01',
                        'inner_split':split[path.name],'width':width,'height':height,'source_sha256':image_sha,
                        'label_sha256':sha(source_label),'derived_label_sha256':sha(derived_label),
                        'metadata':metadata,'boxes':boxes,'port_box_count':len(ports),
                        'tiles':tiles,'uncovered_port_indices':missing})
        kind=path.stem.split('_')[0]
        if kind not in seen_kinds:seen_kinds.add(kind);selected.append(path.name)
        if i%40==0:print(f'train audit {i}/240',flush=True)
    def stats(values):
        a=np.asarray(values,dtype=float)
        return {'count':len(values),'min':float(a.min()),'p10':float(np.quantile(a,.1)),
                'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max())} if len(a) else {'count':0}
    sizes={}
    for cls in (3,4):
        records=[(s,b) for s in samples for b in s['boxes'] if b['source_class']==cls]
        sides=[min(b['box_xyxy'][2]-b['box_xyxy'][0],b['box_xyxy'][3]-b['box_xyxy'][1]) for _,b in records]
        full=[v*960/max(s['width'],s['height']) for (s,b),v in zip(records,sides)]
        tiled=[v*960/1280 for v in sides]
        sizes[str(cls)]={'short_side_original_px':stats(sides),'short_side_full960_px':stats(full),
                         'short_side_tile1280_to960_px':stats(tiled),
                         'full960_below8px':sum(v<8 for v in full),'tiled_below8px':sum(v<8 for v in tiled)}
    partitions={}
    for name in ('inner_train','inner_val'):
        group=[s for s in samples if s['inner_split']==name];tiles=[t for s in group for t in s['tiles']]
        partitions[name]={'images':len(group),'kind_counts':dict(Counter(s['kind'] for s in group)),
            'port_positive_images':sum(s['port_box_count']>0 for s in group),
            'all_tiles':len(tiles),'unsafe_tiles':sum(not t['usable'] for t in tiles),
            'usable_positive_tiles':sum(t['usable'] and bool(t['labels']) for t in tiles),
            'usable_port_negative_tiles':sum(t['usable'] and not t['labels'] for t in tiles),
            'complete_port_label_copies':sum(len(t['labels']) for t in tiles if t['usable'])}
    result={'status':'complete','source_split':'train01','source_images_read':len(samples),'outer_val_images_read':0,
            'test_images_or_labels_read':0,'inference_or_training':False,'source_class_counts':dict(Counter(str(b['source_class']) for s in samples for b in s['boxes'])),
            'port_positive_images':sum(s['port_box_count']>0 for s in samples),'port_negative_images':sum(s['port_box_count']==0 for s in samples),
            'clipped_source_boxes':sum(b['clipped'] for s in samples for b in s['boxes']),
            'derived_labels_exact_match':len(samples),'class_size_stats':sizes,'partitions':partitions,
            'uncovered_port_boxes':sum(len(s['uncovered_port_indices']) for s in samples),
            'exact_duplicate_image_groups':[v for v in hash_groups.values() if len(v)>1],
            'nonempty_exif_metadata_counts':{k:sum(bool(s['metadata'][k]) for s in samples) for k in ('271','272','306','36867','42033')},
            'dimensions':dict(Counter(str((s['width'],s['height'])) for s in samples)),
            'sample_manifest':samples,'elapsed_seconds':time.perf_counter()-start,
            'fingerprints':{str(path):sha(path) for path in (Path(__file__),args.repo/'inspection_agent/port_training_audit.py',args.repo/'inspection_agent/port_tiling.py')},
            'warning':'Source grouping is image-only, not independently verified capture/device grouping; inner image holdout is not cross-scene evidence. Rectangles are not true masks; port-negative is not assembly-normal.'}
    # Four fixed training controls, selected by first filename per kind only.
    board=Image.new('RGB',(1600,1300),'#20252a')
    for index,name in enumerate(selected):
        sample=next(s for s in samples if s['image']==name)
        with Image.open(data/'images/train01'/name) as source:panel=source.convert('RGB')
        draw=ImageDraw.Draw(panel)
        for b in sample['boxes']:
            if b['source_class'] in (3,4):draw.rectangle(b['box_xyxy'],outline='magenta' if b['source_class']==3 else 'cyan',width=6)
        panel.thumbnail((790,590));cell=Image.new('RGB',(800,650),'#20252a');cell.paste(panel,(0,45))
        ImageDraw.Draw(cell).text((8,8),name+' | magenta class3 / cyan class4 | '+sample['inner_split'],fill='white')
        board.paste(cell,((index%2)*800,(index//2)*650))
    board.save(args.output/'train_controls.png')
    result['visual_controls']=selected
    for sample in samples:
        assert sha(data/'images/train01'/sample['image'])==sample['source_sha256']
        assert sha(data/'labels/train01'/(Path(sample['image']).stem+'.txt'))==sample['label_sha256']
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sample_manifest','fingerprints','exact_duplicate_image_groups')},indent=2))


if __name__=='__main__':main()
