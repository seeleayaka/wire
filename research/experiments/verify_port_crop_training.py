"""Read-only source/plan/crop/label replay, no model inference or training."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh verification output required')
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    dataset=json.loads((args.dataset/'dataset_manifest.json').read_text(encoding='utf-8'))
    assert dataset['status']=='complete' and dataset['source_split']=='train01'
    assert dataset['plan_sha256']==sha(args.plan) and dataset['source_group_disjoint']
    assert dataset['training_started'] is False and dataset['outer_val_or_test_source_reads'] is False
    yaml=(args.dataset/'port_crops_rectseg.yaml').read_text(encoding='utf-8')
    assert yaml=='path: '+Path(dataset['final_data_root']).as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n'
    source=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    expected={};source_sets={'train':set(),'val':set()}
    for sample in plan['manifest']:
        split={'inner_train':'train','inner_val':'val'}[sample['inner_split']]
        source_sets[split].add(sample['image'])
        assert sha(source/'images/train01'/sample['image'])==sample['source_sha256']
        assert sha(source/'labels/train01'/(Path(sample['image']).stem+'.txt'))==sample['label_sha256']
        for tile in sample['tiles']:expected[split,sample['image'],tile['tile_id']]=tile
    assert not source_sets['train']&source_sets['val'] and len(source_sets['train'])==192 and len(source_sets['val'])==48
    keys=set();image_files=set();label_files=set()
    for record in dataset['records']:
        key=record['split'],record['source_image'],record['tile_id'];assert key not in keys;keys.add(key)
        tile=expected[key];x,y,r,b=tile['window'];width,height=r-x,b-y
        stem=Path(record['source_image']).stem+'__t'+str(record['tile_id']).zfill(2)
        assert record['image']==f'images/{record["split"]}/{stem}.jpg'
        assert record['label']==f'labels/{record["split"]}/{stem}.txt'
        image_path=args.dataset/record['image'];label_path=args.dataset/record['label']
        assert sha(image_path)==record['image_sha256'] and sha(label_path)==record['label_sha256']
        with Image.open(image_path) as image:assert image.size==(width,height)
        lines=[]
        for label in tile['labels']:
            l,t,rr,bb=label['box_xyxy_local'];assert 0<=l<rr<=width and 0<=t<bb<=height
            l,t,rr,bb=l/width,t/height,rr/width,bb/height;cls=label['class_id']
            lines.append(f'{cls} {l:.9f} {t:.9f} {rr:.9f} {t:.9f} {rr:.9f} {bb:.9f} {l:.9f} {bb:.9f}')
        assert label_path.read_text(encoding='utf-8').splitlines()==lines and record['label_count']==len(lines)
        image_files.add(record['image']);label_files.add(record['label'])
    assert keys==set(expected) and len(keys)==1045
    assert image_files=={p.relative_to(args.dataset).as_posix() for p in (args.dataset/'images').rglob('*.jpg')}
    assert label_files=={p.relative_to(args.dataset).as_posix() for p in (args.dataset/'labels').rglob('*.txt')}
    a=json.loads((args.plan.parent/'report.json').read_text(encoding='utf-8'))
    repeat=args.plan.parent/'crop_plan_repeat.json'
    if repeat.exists():assert sha(repeat)==sha(args.plan)
    initialization=Path('E:/wire_harness_training_bundle/weights/yolov8s-seg.pt')
    result={'source_images_and_labels_verified':240,'crop_hashes_sizes_and_labels_exact':1045,
            'source_groups_disjoint':True,'no_extra_or_missing_crops':True,'inner_val_uses_all_tiles':
            all(len(s['tiles'])==12 for s in plan['manifest'] if s['inner_split']=='inner_val'),
            'crop_plan_repeat_exact':repeat.exists(),'safe_exclusion_plan_uncovered_ports':a['uncovered_port_boxes'],
            'retained_label_plan_uncovered_ports':sum(s['uncovered_complete_ports'] for s in plan['summaries'].values()),
            'initialization_path':str(initialization),'initialization_sha256':sha(initialization),
            'dataset_manifest_sha256':sha(args.dataset/'dataset_manifest.json'),'no_training_or_inference':True}
    assert result['inner_val_uses_all_tiles']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
