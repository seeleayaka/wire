"""Independent source-label/JPEG reproduction of TRAIN-only added crops."""
import hashlib
import io
import json
from pathlib import Path
from collections import Counter
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/allport480_training_20261004'
DATA=Path('E:/PythonProject10/data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def main():
    out=BASE/'dataset_audit'
    if out.exists():
        raise FileExistsError('fresh audit required')
    manifest_path=BASE/'dataset/dataset_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    prior_path=ROOT/'artifacts/port_training_multiscale_20261002/dataset/dataset_manifest.json'
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    old={r['image']:r for r in prior['records'] if r['split']=='train'}
    train_names={r['source_image'] for r in old.values()}
    seen=set()
    counts=Counter()
    for row in manifest['records']:
        assert row['split']=='train' and row['source_image'] in train_names
        assert row['image'] not in seen
        seen.add(row['image'])
        for kind in ('image','label'):
            assert digest(BASE/'dataset'/row[kind])==row[kind+'_sha256']
        if row['image'] in old:
            for k,v in old[row['image']].items():
                assert row[k]==v
            counts['old_rows_byte_identical']+=1
            continue
        assert row['role']=='allport480_positive'
        source=DATA/'images/train01'/row['source_image']
        source_label=DATA/'labels/train01'/(Path(row['source_image']).stem+'.txt')
        assert digest(source)==row['source_sha256']
        with Image.open(source) as image:
            w,h=image.size
            x,y,r,b=row['window']
            assert r-x==b-y==480 and 0<=x<r<=w and 0<=y<b<=h
            encoded=io.BytesIO()
            image.crop((x,y,r,b)).convert('RGB').save(encoded,format='JPEG',quality=95,subsampling=0)
        assert hashlib.sha256(encoded.getvalue()).hexdigest()==row['image_sha256']
        expected=[]
        for line in source_label.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls not in (3,4):
                continue
            left=max(0,cx-bw/2)*w;top=max(0,cy-bh/2)*h
            right=min(1,cx+bw/2)*w;bottom=min(1,cy+bh/2)*h
            if right<=x or left>=r or bottom<=y or top>=b:
                continue
            assert x<=left<right<=r and y<=top<bottom<=b
            assert not ((x>0 and left-x<=16) or (y>0 and top-y<=16)
                        or (r<w and r-right<=16) or (b<h and b-bottom<=16))
            ll,tt,rr,bb=(left-x)/480,(top-y)/480,(right-x)/480,(bottom-y)/480
            expected.append([cls-3,ll,tt,rr,tt,rr,bb,ll,bb])
        actual=[list(map(float,line.split())) for line in (BASE/'dataset'/row['label']).read_text().splitlines() if line.strip()]
        assert len(expected)==len(actual)==row['label_count'] and actual
        for a,e in zip(actual,expected):
            assert len(a)==9 and all(abs(u-v)<=1e-9 for u,v in zip(a,e))
        counts['new_JPEG_byte_reproductions']+=1
        counts['new_source_label_replays']+=1
        counts['new_label_instances']+=len(actual)
    assert counts['old_rows_byte_identical']==656 and counts['new_source_label_replays']==319
    assert len(seen)==975 and len(train_names)==192
    assert all(digest(Path(p))==v for p,v in manifest['protected_pins'].items())
    report=dict(status='complete',counts=dict(counts),manifest_sha256=digest(manifest_path),
        auditor_sha256=digest(Path(__file__)),independent_original_label_decode=True,
        JPEG_bytes_reproduced=True,heldout_images_or_labels_read=False,
        candidate_model_accuracy_evaluated=False,production_changed=False)
    out.mkdir()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
