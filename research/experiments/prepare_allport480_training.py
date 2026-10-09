"""Materialize audited TRAIN-only dataset; retain old labels without edits."""
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(Path('E:/PythonProject10'))]
import allport480_readiness as ready
from prepare_port_training_multiscale import add_crop
from inspection_agent.optional_port_crop_review import sha
from PIL import Image

BASE=ROOT/'artifacts/allport480_training_20261004'
DATASET=BASE/'dataset'


def main():
    if DATASET.exists():
        raise FileExistsError('fresh dataset required')
    report_path=ready.OUT/'report.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status']=='complete' and report['independent_label_geometry_replayed']
    assert report['counts']['safe_windows']==319 and report['counts']['targets']==344
    assert all(sha(Path(p))==v for p,v in report['pins'].items())
    old=json.loads((ready.BASE/'dataset/dataset_manifest.json').read_text(encoding='utf-8'))
    rows=[r for r in old['records'] if r['split']=='train']
    assert len(rows)==656
    for d in ('images/train','labels/train'):
        (DATASET/d).mkdir(parents=True)
    records=[]
    for r in rows:
        for kind in ('image','label'):
            src=ready.BASE/'dataset'/r[kind]
            assert sha(src)==r[kind+'_sha256']
            dst=DATASET/r[kind]
            shutil.copy2(src,dst)
            assert sha(dst)==r[kind+'_sha256']
        records.append(dict(r,retained_byte_identically=True))
    for case in report['cases']:
        name=case['source_image']
        image_path=ready.DATA/'images/train01'/name
        assert sha(image_path)==report['pins'][str(image_path)]
        with Image.open(image_path) as image:
            for i,plan in enumerate(case['windows']):
                assert ready.independent_labels(case['ports'],plan['window'],case['shape'])==plan['labels']
                record=add_crop(DATASET,Path(name).stem+f'__allport480_{i:03}',
                    image.crop(tuple(plan['window'])),plan['labels'],name,'allport480_positive',plan['window'],
                    dict(source_sha256=sha(image_path),seed_source_box_indices=plan['seed_indices']))
                ready.verify_saved_labels(DATASET/record['label'],plan['labels'],plan['window'])
                records.append(record)
    assert len(records)==975 and all(sha(Path(p))==v for p,v in report['pins'].items())
    assert len({r['source_image'] for r in records})==192
    manifest=dict(status='complete',train_sources=192,train_crops=975,records=records,
        readiness_sha256=sha(report_path),plan_sha256=sha(BASE/'PLAN.md'),
        script_sha256=sha(Path(__file__)),protected_pins=report['pins'],
        old_records_preserved=656,new_crops=319,new_label_geometry_replayed=True,
        heldout_images_or_labels_materialized=False,training_started=False)
    (DATASET/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('records','protected_pins')},indent=2))


if __name__=='__main__':
    main()
