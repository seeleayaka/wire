"""Replay every frozen port hint from cached predictions; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    sys.path.insert(0,str(args.repo))
    from inspection_agent.port_state_hint import select_port_state_hints
    report=json.loads(args.report.read_text(encoding='utf-8'))
    assert report['status']=='complete' and report['split']=='val01'
    assert len(report['calibration'])==30 and len(report['cases'])==30
    assert all(c['image'].startswith('normal_') for c in report['calibration'])
    threshold=max(.25,float(np.quantile([c['maximum_confidence'] for c in report['calibration']],.95)))
    assert threshold==report['threshold']
    for item in report['calibration']:
        assert item['maximum_confidence']==max((r['confidence'] for r in item['predictions']),default=0.)
    frozen=json.loads((args.repo/'output/mendeley_cnn_qualified_support_20260929/report.json').read_text(encoding='utf-8'))
    parents={c['image']:c['candidates'] for c in frozen['results']['cnn_calibrated_support']['cases']}
    assert set(parents)=={c['image'] for c in report['cases']}
    for case in report['cases']:
        assert case['parents']==parents[case['image']] and case['split']=='val01'
        replay=select_port_state_hints(case['parents'],case['aligned_predictions'],threshold)
        assert replay=={k:case[k] for k in ('parents','hints','selection_audit')}
        # Exact output under reversed prediction order, excluding traversal audit counts.
        reversed_replay=select_port_state_hints(case['parents'],list(reversed(case['aligned_predictions'])),threshold)
        assert reversed_replay['hints']==replay['hints']
        assert len(case['source_classes'])==len(case['targets'])
    data=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
    assert len(report['source_manifest'])==60
    for item in report['source_manifest']:
        assert item['split'] in ('train01','val01')
        assert sha(data/item['split']/item['image'])==item['source_sha256']
    for path,digest in report['source_hashes'].items():assert sha(path)==digest,path
    result={'cases_exact_replay':30,'calibration_rebuilt_exact':True,'parents_unchanged':True,
            'hints_order_stable':True,'source_fingerprints_verified':60,'no_model_inference':True,
            'extra_precise_fragments':report['with_hints']['iou_ge_05']-report['baseline']['iou_ge_05'],
            'additional_normal_hints':report['with_hints']['normal_regions']-report['baseline']['normal_regions']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
