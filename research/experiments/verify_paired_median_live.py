"""Fresh initial review + current paired + median + original reference gates."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE, read_targets
from audit_port_multiscale_acceptance import metric, matches
from paired_graph_hint_link import accepted_native_rows
from paired_geometry_live_contract import validate_upstream
from paired_median_geometry_backend import append_median_review
from inspection_agent.paired_port_geometry import run_paired_geometry_review, paired_runtime_fingerprint
from inspection_agent.optional_port_crop_review import SCENE
from inspection_agent.context_port_recheck import render_consensus_overlay
from verify_paired_geometry_live import parity
TRIAL = ROOT / 'artifacts/paired_median_current_head_20261003'
OUT = ROOT / 'artifacts/paired_median_live_20261003'
os.environ.update(QT_QPA_PLATFORM='offscreen', HF_HUB_OFFLINE='1', YOLO_OFFLINE='True', YOLO_AUTOINSTALL='False',
    YOLO_CONFIG_DIR=str(OUT / 'yolo_config'), PYTHONPROJECT10_DINO_CACHE=str(OUT / 'dino_cache'))


def main():
    if OUT.exists(): raise FileExistsError('Preserve real median review evidence')
    audit_path = ROOT / 'artifacts/paired_median_source_audit_20261003/report.json'
    audit = load(audit_path); assert audit['source_gates_passed']; pins = dict(audit['pins'])
    assert {p: sha(Path(p)) for p in pins} == pins
    frozen = paired_runtime_fingerprint(REPO); categories = {}; snapshots = {}; entries = {}
    def add(stage, name, reason): categories.setdefault((stage, name), []).append(reason)
    for row in audit['cases']:
        if row['additions']: add(row['stage'], row['image'], 'ALL_new_median_candidates')
    for stage in ('train', 'inner', 'outer'):
        entries[stage] = {r['image']: r for r in load(BASE / stage / 'report.json')['cases']}
        directory = GEOMETRY / 'full_train' if stage == 'train' else GEOMETRY / 'holdouts' / stage
        for row in load(directory / 'report.json')['cases']:
            if row['additions']: add(stage, row['image'], 'ALL_current_paired_candidates')
        add(stage, sorted(n for n in entries[stage] if n.startswith('normal_'))[0], 'first_normal_control')
    controls = ROOT / 'artifacts/port_extended_training_controls_20261003/report.json'; pins[str(controls)] = sha(controls)
    for row in load(controls)['cases']:
        if row['metrics']['current']['predictions'] > 0: add('train', row['image'], 'ALL_old_cue_controls')
    for path, stage in ((ROOT / 'artifacts/paired_port_semantics_20261003/holdouts/inner/report.json', 'inner'),
                        (ROOT / 'artifacts/paired_semantic_committee_20261003/outer/report.json', 'outer')):
        pins[str(path)] = sha(path)
        for row in load(path)['cases']:
            if row['trial']['unmatched'] > row['current']['unmatched']: add(stage, row['image'], 'ALL_previous_rejected_risk_sources')
    for stage, name in categories:
        path = TRIAL / stage / (Path(name).stem + '_predictions.json'); pins[str(path)] = sha(path); snapshots[stage, name] = load(path)
        source = DATA / 'images' / ('val01' if stage == 'outer' else 'train01') / name; pins[str(source)] = sha(source)
    reference = DATA / 'images/train01/normal_073.JPG'
    for path in (Path(__file__), Path(__file__).with_name('paired_median_geometry_backend.py'),
        Path(__file__).with_name('paired_median_proposals.py'), Path(__file__).with_name('paired_geometry_live_contract.py'),
        ROOT / 'artifacts/paired_median_live_preregistration_20261003/PLAN.md', reference): pins[str(path)] = sha(path)
    OUT.mkdir(); (OUT / 'yolo_config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf', OUT / 'yolo_config/Ultralytics/Arial.ttf')
    save(OUT / 'protocol.json', dict(pins=pins, accepted_paired_runtime=frozen,
        categories=[dict(stage=s, image=n, reasons=r) for (s, n), r in categories.items()],
        fresh_initial_review=True, GT_after_final_hint_selection=True, no_automatic_deployment=True, sam_pending=True, field_accuracy=False))
    import torch
    torch.set_num_threads(4)
    import cv2
    import numpy as np
    import assembly_auto_review_dino_v2 as entry
    gui = entry.implementation; app = gui.QApplication.instance() or gui.QApplication([])
    previous = gui.adaptive.robust.auto.base.OUT; started = time.monotonic(); records = []
    def progress(**fields):
        record = dict(status='running', pid=os.getpid(), seconds=round(time.monotonic() - started, 2)); record.update(fields)
        save(OUT / 'progress.json', record)
    try:
        for index, ((stage, name), reasons) in enumerate(categories.items()):
            target = OUT / stage / Path(name).stem; target.mkdir(parents=True); gui.adaptive.robust.auto.base.OUT = target
            source = DATA / 'images' / ('val01' if stage == 'outer' else 'train01') / name
            saved = snapshots[stage, name]; expected_candidates = saved['trial']['paired_semantic_additions']
            progress(stage=stage, image=name, completed=index, total=len(categories), phase='fresh_initial_SIFT_DINO')
            cv2.setRNGSeed(0); worker = gui.InitialReviewWorker(reference, source, [[.03, .04, .97, .96]])
            payloads = []; worker.completed.connect(payloads.append); worker.run(); assert len(payloads) == 1
            payload = payloads[0]; assert payload['status'] == 'ready_for_sam3', payload['status']
            report = payload['report']; directory = Path(payload['output']); save(directory / 'initial_report.json', report)
            protected = copy.deepcopy(report)
            progress(stage=stage, image=name, completed=index, total=len(categories), phase='actual_current_accepted_paired')
            original = run_paired_geometry_review(report, project=REPO, paired_enabled=True, enabled=True, scene=SCENE,
                supplementary_enabled=True, student_enabled=True, feature_enabled=True, resolution_enabled=True)
            upstream = validate_upstream(report, original, expected_candidates, normal_control=name.startswith('normal_'))
            save(directory / 'current_paired_ports.json', original)
            progress(stage=stage, image=name, completed=index, total=len(categories), phase='fresh_median_and_original_reference_gates')
            output = append_median_review(report, original, project=REPO); assert report == protected
            save(directory / 'median_ports.json', output)
            assert not output['median_geometry_policy'].get('fallback_reason'), output['median_geometry_policy']
            evidence = output.get('median_geometry_evidence'); accepted = []
            if evidence:
                parity(saved['current']['all_predictions'], evidence['native_current']['all_predictions'])
                parity(expected_candidates, evidence['native']['paired_semantic_additions'])
                accepted = accepted_native_rows(evidence['native']['paired_semantic_additions'],
                    output['supplementary_hints'][len(original['supplementary_hints']):],
                    np.asarray(report['alignment']['source_to_reference_homography']), [2736, 3648], [2736, 3648])
            else: assert not expected_candidates, 'Median candidate cannot silently abstain'
            for key, value in original.items():
                if key == 'supplementary_hints':
                    assert output[key][:len(value)] == value
                else: assert output[key] == value
            assert len(output['rescue_hints']) <= 5 and len(output['supplementary_hints']) <= 5
            render_consensus_overlay(directory / 'aligned.jpg', output, directory / 'median_overlay.jpg')
            selected = saved['current']['all_predictions'] + accepted
            targets = read_targets(stage, name, [2736, 3648], entries[stage][name]['label_sha256'], pins)
            oh, nh = matches(saved['current']['all_predictions'], targets)[0], matches(selected, targets)[0]
            row = dict(stage=stage, image=name, reasons=reasons, upstream_path=upstream,
                current=metric(saved['current']['all_predictions'], targets), trial=metric(selected, targets),
                gained=sorted(nh - oh), lost=sorted(oh - nh), expected_candidates=len(expected_candidates), accepted=len(accepted),
                initial_report=str(directory / 'initial_report.json'), current_evidence=str(directory / 'current_paired_ports.json'),
                evidence=str(directory / 'median_ports.json'), overlay=str(directory / 'median_overlay.jpg'), sam_pending=True)
            records.append(row); save(OUT / 'partial.json', dict(cases=records)); print(str(row), flush=True)
            assert not row['lost'] and row['trial']['unmatched'] <= row['current']['unmatched']
            assert len(accepted) == len(expected_candidates), 'Safety gate removed candidate; not a live pass'
            if name.startswith('normal_'): assert not selected
        gains = {stage: sum(r['trial']['tp'] - r['current']['tp'] for r in records if r['stage'] == stage) for stage in ('train', 'inner', 'outer')}
        assert gains == dict(train=3, inner=1, outer=0)
        assert {p: sha(Path(p)) for p in pins} == pins and paired_runtime_fingerprint(REPO) == frozen
        result = dict(status='complete', qualifies_live_diagnostic=True, cases=records, gains=gains,
            safety_abstentions=sum(r['upstream_path'] != 'applied' for r in records),
            seconds=round(time.monotonic() - started, 2), sam_pending=True, validation_reused=True, field_accuracy=False)
        save(OUT / 'report.json', result); progress(status='complete', gains=gains, qualifies_live_diagnostic=True)
    except BaseException as error:
        progress(status='failed', error=type(error).__name__ + ': ' + str(error)); raise
    finally:
        gui.adaptive.robust.auto.base.OUT = previous; app.processEvents()


if __name__ == '__main__': main()
