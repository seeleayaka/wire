"""Materialize a frozen train-only crop plan into a new rect-seg dataset."""
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
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--final-data-root',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh dataset required')
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    assert plan['status']=='complete' and plan['source_split']=='train01' and not plan['outer_val_or_test_source_reads']
    for path,digest in plan['fingerprints'].items():assert sha(Path(path))==digest
    image_dir=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images/train01'
    label_dir=image_dir.parents[1]/'labels/train01'
    records=[];args.output.mkdir(parents=True)
    for sample in plan['manifest']:
        split={'inner_train':'train','inner_val':'val'}[sample['inner_split']]
        image_path=image_dir/sample['image'];assert sha(image_path)==sample['source_sha256']
        assert sha(label_dir/(Path(sample['image']).stem+'.txt'))==sample['label_sha256']
        with Image.open(image_path) as source:
            for tile in sample['tiles']:
                x,y,r,b=tile['window'];width,height=r-x,b-y
                assert 0<=x<r<=source.width and 0<=y<b<=source.height
                stem=Path(sample['image']).stem+'__t'+str(tile['tile_id']).zfill(2)
                image_target=args.output/'images'/split/(stem+'.jpg')
                label_target=args.output/'labels'/split/(stem+'.txt')
                image_target.parent.mkdir(parents=True,exist_ok=True);label_target.parent.mkdir(parents=True,exist_ok=True)
                # JPEG at fixed quality: derived training pixels, never overwrite source.
                source.crop((x,y,r,b)).convert('RGB').save(image_target,quality=95,subsampling=0)
                lines=[]
                for label in tile['labels']:
                    l,t,rr,bb=label['box_xyxy_local'];assert 0<=l<rr<=width and 0<=t<bb<=height
                    cls=label['class_id'];assert cls in (0,1)
                    l,t,rr,bb=l/width,t/height,rr/width,bb/height
                    lines.append(f'{cls} {l:.9f} {t:.9f} {rr:.9f} {t:.9f} {rr:.9f} {bb:.9f} {l:.9f} {bb:.9f}')
                label_target.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
                with Image.open(image_target) as reread:assert reread.size==(width,height)
                assert label_target.read_text(encoding='utf-8').splitlines()==lines
                records.append({'source_image':sample['image'],'split':split,'tile_id':tile['tile_id'],
                                'image':image_target.relative_to(args.output).as_posix(),
                                'label':label_target.relative_to(args.output).as_posix(),
                                'image_sha256':sha(image_target),'label_sha256':sha(label_target),'label_count':len(lines)})
        assert sha(image_path)==sample['source_sha256']
    yaml=args.output/'port_crops_rectseg.yaml'
    yaml.write_text('path: '+args.final_data_root.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: unplugged_plug\n  1: unplugged_jack\n',encoding='utf-8')
    assert len(records)==1045 and sum(r['split']=='train' for r in records)==469
    result={'status':'complete','source_split':'train01','plan_sha256':sha(args.plan),
            'source_images':240,'train_crops':469,'inner_val_crops':576,'source_group_disjoint':
            not ({r['source_image'] for r in records if r['split']=='train'} & {r['source_image'] for r in records if r['split']=='val'}),
            'final_data_root':str(args.final_data_root),'jpeg_quality':95,'jpeg_subsampling':0,
            'training_started':False,'outer_val_or_test_source_reads':False,'records':records,
            'warning':'Rectangular segmentation labels, not true masks. Training-only label-based sampling; validation tile selection is label-independent. Derived JPEGs are lossy.'}
    assert result['source_group_disjoint']
    (args.output/'dataset_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
