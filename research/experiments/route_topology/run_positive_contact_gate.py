"""Frozen FIT-only positive-contact rescue gate, never electrical continuity."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from component_cues import gate

ROOT = Path(__file__).resolve().parents[2]

def bright_contacts(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 1] < 80) & (hsv[:, :, 2] >= 160)

def main():
    out = ROOT / 'artifacts/mendeley_positive_contact_source_20261006'
    out.mkdir(exist_ok=False)
    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    labels_path = ROOT / 'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    split_path = ROOT / 'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json'
    rescue_path = ROOT / 'artifacts/mendeley_component_rescue_source_20261006/report.json'
    baseline_path = ROOT / 'artifacts/mendeley_component_color_source_20261006/report.json'
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    byid = {r['id']: r for r in labels['rows']}
    before = source_pins()
    pins = {str(p): sha256(p) for p in [Path(__file__), labels_path, split_path, rescue_path, baseline_path]}
    save('protocol.json', dict(pins=pins, mainline_pins=before,
        contact_mask='S<80 and V>=160, native registered pixels; no gap closing',
        fit_support='FIT exposed occupancy>=.6 and exposed-minus-visible>=.4; minimum6 pixels',
        threshold='max(FIT visible95th percentile,FIT exposed5th percentile); strict greater',
        rule='preserve old singletons; new exposed rescue requires positive contact support',
        fixed_before_current_calibration_replay=True, designed_after_prior_source_failures=True,
        source_labels_not_human_GT=True, repeated_development_not_field_accuracy=True,
        existing_reviewed_poses_reused=True, originals_decoded_fresh=True,
        no_inspection_images_read=True, no_parameter_sweep=True))
    masks = {}
    for identity in split['fit_ids'] + split['calibration_ids']:
        row = byid[identity]
        if sha256(row['source_path']) != row['source_sha256']:
            raise ValueError('source drift')
        rgb = np.asarray(Image.open(row['source_path']).convert('RGB'))
        matrix = np.array([[1, 0, -1600], [0, 1, -1000], [0, 0, 1]]) @ np.array(row['pose']['inspection_to_reference_local'])
        patch = cv2.warpPerspective(rgb, matrix, (100, 50), flags=cv2.INTER_LINEAR)
        if not np.array_equal(patch, np.asarray(Image.open(row['patch_path']).convert('RGB'))):
            raise ValueError('reviewed patch mismatch')
        masks[identity] = bright_contacts(patch)
    fit = split['fit_ids']
    exposed = np.stack([masks[i] for i in fit if byid[i]['visual_label'] == 0])
    visible = np.stack([masks[i] for i in fit if byid[i]['visual_label'] == 1])
    support = (exposed.mean(0) >= .6) & (exposed.mean(0) - visible.mean(0) >= .4)
    Image.fromarray(support.astype(np.uint8)*255).save(out / 'fit_contact_support.png')
    support_count = int(support.sum())
    scores = {i: float(m[support].mean()) if support_count else 0.0 for i, m in masks.items()}
    threshold = max(float(np.quantile([scores[i] for i in fit if byid[i]['visual_label']==1], .95)),
                    float(np.quantile([scores[i] for i in fit if byid[i]['visual_label']==0], .05)))
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))['baseline_LOO_results']
    proposals = {r['id']: r for r in json.loads(rescue_path.read_text(encoding='utf-8'))['cases']}
    rows = []; gains = []; losses = []
    for old in baseline:
        i = old['id']; old_label = old['prediction']['visual_label_candidate']
        proposed = proposals[i]['prediction']['visual_label_candidate']
        positive = support_count >= 6 and scores[i] > threshold
        accepted = old_label if old_label is not None else (proposed if proposed == 0 and positive else None)
        rows.append(dict(id=i, visual_label=old['visual_label'], positive_contact_score=scores[i],
                         positive_contact_supported=positive, prediction=dict(visual_label_candidate=accepted)))
        if old_label is None and accepted == old['visual_label']: gains.append(i)
        if old_label == old['visual_label'] and accepted != old_label: losses.append(i)
    result = gate(rows)
    if before != source_pins() or any(sha256(p) != digest for p, digest in pins.items()):
        raise ValueError('E/input/code drift')
    save('report.json', dict(status='complete', support_pixels=support_count, fit_threshold=threshold,
        fit_prerequisite_passed=support_count>=6, source_gate=result, source_gains=gains, source_losses=losses,
        strict_net_source_gain=support_count>=6 and result['passed'] and bool(gains) and not losses,
        cases=rows, mainline_unchanged=True, deployed=False, new_confirmed_connections=0,
        contact_mask_not_certified_metal_identity=True, repeated_development_not_field_accuracy=True))
    print(json.dumps(dict(support_pixels=support_count, threshold=threshold, source_gate=result, gains=gains, losses=losses)))

if __name__ == '__main__': main()
