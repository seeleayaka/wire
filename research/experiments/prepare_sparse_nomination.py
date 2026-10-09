"""Finite ALL192 GT-free nomination feasibility; no detector or SAM inference."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, read_image, REFERENCE_SHA
from inspection_agent.paired_native_pose import HEAD_RELATIVE, HEAD_SHA, native_pose_runtime_fingerprint
from inspection_agent.paired_port_features import expected_in_source, valid_boxes, embeddings, paired_features
from sparse_voter_nomination import nominees, actions

PREP = ROOT / 'artifacts/two_vote_resolution_preparation_20261004'
CONTROL = ROOT / 'artifacts/roi_checkpoint_completion_20261004/report.json'
CONTROL_AUDIT = ROOT / 'artifacts/roi_checkpoint_completion_replay_20261004/report.json'
ALIGN = ROOT / 'artifacts/paired_port_semantics_20261003/features_train'
OUT = ROOT / 'artifacts/sparse_nomination_20261004'
PLAN = ROOT / 'artifacts/sparse_nomination_preregistration_20261004/PLAN.md'


def main():
    import cv2
    import torch
    import psutil
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError('Preserve nomination feasibility')
    assert psutil.virtual_memory().available > 6 * 2**30
    control, audited, prep = load(CONTROL), load(CONTROL_AUDIT), load(PREP/'report.json')
    assert control['status'] == 'rejected' and audited['status'] == 'pass'
    assert audited['source_report_sha256'] == sha(CONTROL)
    frozen = native_pose_runtime_fingerprint(REPO)
    assert control['runtime'] == prep['runtime'] == frozen
    assert all(sha(Path(p)) == v for p, v in control['pins'].items())
    assert all(sha(Path(p)) == v for p, v in prep['pins'].items())
    referencepath = DATA/'images/train01/normal_073.JPG'
    assert sha(referencepath) == REFERENCE_SHA and sha(REPO/HEAD_RELATIVE) == HEAD_SHA
    pins = {str(p): sha(p) for p in (Path(__file__), PLAN, CONTROL, CONTROL_AUDIT, PREP/'report.json',
            referencepath, REPO/HEAD_RELATIVE, Path(__file__).with_name('sparse_voter_nomination.py'),
            Path(__file__).with_name('unresolved_voter_seeds.py'), Path(__file__).with_name('novel_box_geometry.py'),
            REPO/'inspection_agent/paired_port_features.py')}
    OUT.mkdir()
    folder = OUT/'train'
    folder.mkdir()
    start = time.monotonic()
    head = torch.nn.Linear(6144, 3)
    head.load_state_dict(torch.load(REPO/HEAD_RELATIVE, map_location='cpu', weights_only=True)['state_dict'])
    head.eval().requires_grad_(False)
    reference = read_image(referencepath)
    encoder = None
    rows = []
    save(OUT/'protocol.json', dict(pins=pins, runtime=frozen, no_GT_read=True,
         no_new_detector_inference=True, no_SAM=True, nomination_probability_gate=.98,
         nomination_distinct_votes=[1, 2], final_fault_votes_required=3,
         no_deployment=True, field_accuracy=False))
    try:
        for index, record in enumerate(prep['cases']):
            assert time.monotonic()-start < 7200, 'Finite nomination preparation deadline'
            name = record['image']
            save(OUT/'progress.json', dict(status='running', pid=os.getpid(), image=name,
                 completed=index, total=192, seconds=round(time.monotonic()-start, 2), phase='frozen_action_nomination'))
            path = Path(record['path'])
            assert sha(path) == record['sha256']
            pins[str(path)] = record['sha256']
            case = load(path)
            seeds = nominees(case['models'], case['current'], case['source_sha256'], (2736, 3648)) if case['remaining_budget'] else []
            raw_count = len(seeds)
            scores, chosen, alignment, skip = [], [], None, None
            source = DATA/'images/train01'/name
            pins[str(source)] = sha(source)
            assert pins[str(source)] == case['source_sha256']
            if seeds:
                alignpath = ALIGN/(Path(name).stem+'_source.json')
                pins[str(alignpath)] = sha(alignpath)
                aligned = load(alignpath)
                assert aligned['source_sha256'] == case['source_sha256']
                alignment = aligned['alignment']
                if not alignment.get('alignment_quality', {}).get('reliable'):
                    seeds = []
                    skip = 'original_registration_abstention'
                else:
                    image = read_image(source)
                    assert image.shape[:2] == (2736, 3648)
                    expected, mask = expected_in_source(reference, alignment['source_to_reference_homography'], image.shape[:2])
                    seeds = [seeds[i] for i in valid_boxes([s['box_xyxy'] for s in seeds], mask)]
                    if seeds:
                        if encoder is None:
                            import dino_feature_diff as dino
                            encoder = dino._model()
                            encoder.eval().requires_grad_(False)
                            torch.set_num_threads(2)
                        boxes = [s['box_xyxy'] for s in seeds]
                        vectors = paired_features(embeddings(encoder, image, boxes), embeddings(encoder, expected, boxes))
                        with torch.inference_mode():
                            scores = head(vectors).softmax(1).tolist()
                        featurepath = folder/(Path(name).stem+'_features.pt')
                        torch.save(dict(features=vectors, boxes=boxes, source_sha256=case['source_sha256']), featurepath)
                        pins[str(featurepath)] = sha(featurepath)
                        chosen = actions(seeds, scores, case['remaining_budget'])
            output = folder/(Path(name).stem+'_nominations.json')
            save(output, dict(image=name, source_sha256=case['source_sha256'], current=case['current'],
                 models=case['models'], remaining_budget=case['remaining_budget'], raw_nominees=raw_count,
                 seeds=seeds, probabilities=scores, chosen_seeds=chosen, alignment=alignment,
                 skip_reason=skip, head_sha256=HEAD_SHA, no_final_fault_cues=True))
            pins[str(output)] = sha(output)
            rows.append(dict(image=name, raw_nominees=raw_count, valid_nominees=len(seeds),
                             actions=len(chosen), path=str(output), sha256=pins[str(output)]))
        assert len(rows) == 192
        assert all(sha(Path(p)) == v for p, v in pins.items())
        assert native_pose_runtime_fingerprint(REPO) == frozen
        result = dict(status='complete', sources=192, raw_nominees=sum(r['raw_nominees'] for r in rows),
             valid_nominees=sum(r['valid_nominees'] for r in rows), actions=sum(r['actions'] for r in rows),
             sources_with_actions=sum(r['actions'] > 0 for r in rows), cases=rows, pins=pins, runtime=frozen,
             no_GT_read=True, no_new_detector_inference=True, no_deployment=True, field_accuracy=False,
             next_step='Independent replay and TRAIN coverage; no automatic inference or deployment',
             seconds=round(time.monotonic()-start, 2))
        save(OUT/'report.json', result)
        save(OUT/'progress.json', {k: v for k, v in result.items() if k not in ('pins', 'runtime', 'cases')})
        print({k: v for k, v in result.items() if k not in ('pins', 'runtime', 'cases')}, flush=True)
    except BaseException as error:
        save(OUT/'progress.json', dict(status='failed', error=type(error).__name__+': '+str(error), seconds=round(time.monotonic()-start, 2)))
        raise


if __name__ == '__main__':
    main()
