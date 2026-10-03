"""Verified train-only crop plan; labels choose training samples, never inference samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    sys.path.insert(0,str(args.repo))
    from inspection_agent.port_training_audit import parse_boxes,rectangle_labels,inner_split,plan_tiles
    from inspection_agent.port_crop_labels import labeled_tiles,training_tiles
    audit=json.loads(args.audit.read_text(encoding='utf-8'));assert audit['status']=='complete' and audit['source_split']=='train01'
    data=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    samples=audit['sample_manifest'];assert len(samples)==240
    for path,digest in audit['fingerprints'].items():assert sha(Path(path))==digest,path
    assert set(s['image'] for s in samples)=={p.name for p in (data/'images/train01').glob('*.JPG')}
    partitions=inner_split([s['image'] for s in samples]);manifest=[]
    for s in samples:
        assert s['source_split']=='train01' and s['inner_split']==partitions[s['image']]
        source_label=data/'labels/train01'/(Path(s['image']).stem+'.txt')
        assert sha(data/'images/train01'/s['image'])==s['source_sha256']
        assert sha(source_label)==s['label_sha256']
        boxes=parse_boxes(source_label.read_text(encoding='utf-8'),s['width'],s['height']);assert boxes==s['boxes']
        derived=args.repo/'data/derived/mendeley_port_state_rectseg_20260825/labels/train'/(Path(s['image']).stem+'.txt')
        assert sha(derived)==s['derived_label_sha256'] and rectangle_labels(boxes)==derived.read_text(encoding='utf-8').splitlines()
        assert plan_tiles(boxes,s['width'],s['height'])==s['tiles']
        tiles=labeled_tiles(boxes,s['width'],s['height'])
        selected=training_tiles(s['image'],tiles) if s['inner_split']=='inner_train' else tiles
        assert selected==(training_tiles(s['image'],list(reversed(tiles))) if s['inner_split']=='inner_train' else tiles)
        represented={p['source_box_index'] for t in selected for p in t['labels'] if p['complete']}
        port_ids={i for i,b in enumerate(boxes) if b['source_class'] in (3,4)}
        manifest.append({'image':s['image'],'kind':s['kind'],'source_sha256':s['source_sha256'],
                         'label_sha256':s['label_sha256'],'inner_split':s['inner_split'],
                         'tiles':selected,'uncovered_complete_ports':sorted(port_ids-represented)})
    summaries={}
    for split in ('inner_train','inner_val'):
        group=[s for s in manifest if s['inner_split']==split];tiles=[t for s in group for t in s['tiles']]
        summaries[split]={'source_images':len(group),'tiles':len(tiles),
                          'positive_tiles':sum(bool(t['labels']) for t in tiles),'negative_tiles':sum(not t['labels'] for t in tiles),
                          'label_copies':sum(len(t['labels']) for t in tiles),
                          'complete_label_copies':sum(p['complete'] for t in tiles for p in t['labels']),
                          'partial_label_copies':sum(not p['complete'] for t in tiles for p in t['labels']),
                          'uncovered_complete_ports':sum(len(s['uncovered_complete_ports']) for s in group)}
    dates=Counter((s['metadata']['271'],s['metadata']['272'],s['metadata']['306'][:10]) for s in samples)
    result={'status':'complete','source_split':'train01','outer_val_or_test_source_reads':False,
            'source_and_label_verification':240,'original_safe_exclusion_plan_exact_replay':240,
            'summaries':summaries,'manifest':manifest,'source_audit_sha256':sha(args.audit),
            'metadata_date_groups':[{'make':k[0],'camera':k[1],'modification_date':k[2],'images':v} for k,v in sorted(dates.items())],
            'fingerprints':{str(path):sha(path) for path in (Path(__file__),args.repo/'inspection_agent/port_crop_labels.py')},
            'training_not_started':True,'crop_images_not_written':True,
            'policy':'Training: all labeled tiles, including clipped labels, plus one deterministic port-negative tile/source. Inner validation: all 12 tiles/source, independent of labels.',
            'warning':'Modification-time EXIF is not verified capture identity. Image-grouped inner split is not device-independent. Validation must merge source-image detections and apply the frozen edge rule; clipped training rectangles are not true masks.'}
    assert summaries['inner_train']['uncovered_complete_ports']==0 and summaries['inner_val']['uncovered_complete_ports']==0
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('manifest','fingerprints')},indent=2))


if __name__=='__main__':main()
