"""Verify actual smoke events, parameter change, data routing and stored weight hashes."""
import argparse
import hashlib
import json
import math
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
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh verification output required')
    sys.path.insert(0,str(args.repo))
    from inspection_agent.port_crop_training import BASE_SHA256,training_options
    r=json.loads(args.report.read_text(encoding='utf-8'));assert r['status']=='complete' and r['mode']=='smoke'
    expected=training_options(r['options']['data'],Path(r['options']['project']).parent,'smoke')
    assert r['options']==expected and r['completed_epochs']==1 and not r['outer_validation_or_test_used']
    assert r['initialization_sha256']==BASE_SHA256==sha(Path('E:/wire_harness_training_bundle/weights/yolov8s-seg.pt'))
    manifest=args.repo/'data/derived/port_crop_training_20260929/dataset_manifest.json'
    assert sha(manifest)==r['dataset_manifest_sha256']
    for path,digest in r['fingerprints'].items():assert sha(Path(path))==digest,path
    events=[json.loads(line) for line in (args.report.parent/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    steps=[e for e in events if e['event']=='optimizer_step'];batches=[e for e in events if e['event']=='batch']
    assert len(steps)==r['nonzero_gradient_steps']>0 and len(batches)==4
    assert all(math.isfinite(e['raw_gradient_l2']) and e['raw_gradient_l2']>0 for e in steps)
    assert all(math.isfinite(v) for e in batches for v in e['losses'].values())
    assert r['changed_trainable_parameters']>0 and r['finite_gradients']
    selections=r['smoke_selection'];all_records=[v for records in selections.values() for v in records]
    assert len(selections['train'])==4 and len(selections['val'])==2
    assert not ({v['source_image'] for v in selections['train']}&{v['source_image'] for v in selections['val']})
    for record in all_records:
        for key in ('image','label'):
            assert sha(args.report.parent/'smoke_dataset'/record[key])==record[key+'_sha256']
    for name,metadata in r['weights'].items():
        assert sha(Path(metadata['path']))==metadata['sha256']
        assert sha(args.report.parent/'runs/rectports/weights'/name)==metadata['sha256']
    assert sha(args.report.parent/'config/Ultralytics/Arial.ttf')==sha(Path('C:/Windows/Fonts/arial.ttf'))
    result={'smoke_complete':True,'finite_loss_batches':4,'positive_gradient_steps':len(steps),
            'changed_trainable_parameters':r['changed_trainable_parameters'],'sampled_max_rss_GiB':r['sampled_max_rss_GiB'],
            'six_smoke_crop_pairs_hash_verified':True,'base_and_dataset_hashes_verified':True,'local_font_verified':True,
            'full_recipe_remains_six_epochs_nbs64':training_options('data','out','full')['nbs']==64,
            'no_model_inference':True,'accuracy_gain_not_tested':True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
