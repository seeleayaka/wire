"""Independent native-pixel replay; retains every fragment and old decision gate."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from audit_semantic_visible_bundle_native import flood_components
from core import sha256
from run_audit import source_pins
from run_prompt_contrast import save, verify

ROOT = Path(__file__).resolve().parents[2]

def main():
    evidence = ROOT / 'artifacts/mendeley_typed_affine_bundle_20261006'
    report_path = ROOT / 'artifacts/mendeley_typed_affine_bundle_review_20261006/report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    protocol = json.loads((evidence / 'protocol.json').read_text(encoding='utf-8'))
    prepared = json.loads((evidence / 'preparation_report.json').read_text(encoding='utf-8'))['cases']
    assert report['status'] == 'complete'
    verify(protocol['pins'])
    assert source_pins() == protocol['mainline_pins']
    scope = json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    observed = {'reference': report['reference_observation'], **{c['id']: c['observation'] for c in report['cases']}}
    comparisons = {c['id']: c['comparison'] for c in report['cases']}
    checks = []
    for case in prepared:
        original = case['original_source']; context = case['crop_context']
        assert sha256(original['path']) == original['image_sha256']
        view = observed[case['id']]
        assert view['source_binding']['image_sha256'] == original['image_sha256']
        assert case['sam_inference_requested'] and view['socket_state'] == case['phenotype']
        actual = np.asarray(Image.open(original['path']).convert('RGB').crop(context['crop_box_xyxy']))
        assert np.array_equal(actual, np.asarray(Image.open(context['source']['path']).convert('RGB')))
        recorded = {r['record_id']: r for r in view['mask_audit']}
        eligible = 0; high_touch = False; socket_pixels = []; masks = 0; components = 0
        for recipe in protocol['recipes']:
            run = evidence / case['id'] / recipe
            manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
            for filename, digest in manifest['verified_files'].items():
                assert sha256(filename) == digest
            if recipe != 'cable_plus_reference_anatomy_box': continue
            raw_report = json.loads((run / 'sam/report.json').read_text(encoding='utf-8'))
            assert len(recorded) == len(raw_report['scores'])
            for index, score in enumerate(raw_report['scores'], 1):
                identity = f'mask_{index:03d}'; old = recorded[identity]
                raw = np.asarray(Image.open(run / 'sam' / (identity + '.png')).convert('L')) > 0
                parts = flood_components(raw); masks += 1; components += len(parts)
                ys, xs = np.where(raw); h, w = raw.shape
                truncated = bool(len(xs) and (xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w-2 or ys.max() >= h-2))
                assert truncated == old['boundary_truncated'] and score == old['score']
                signatures = []
                for part in parts:
                    hits = {a['id']: 0 for a in scope['anchors']}
                    for anchor in scope['anchors']:
                        pose = next(p for p in case['anchors'] if p['id'] == anchor['id'])
                        assert pose['localization_proposal_supported'] and all(pose['gates'].values())
                        matrix = np.asarray(pose['inspection_to_reference_local'])
                        l, t, r, b = anchor['bbox_xyxy']
                        for flat in part:
                            y, x = divmod(flat, w)
                            x += context['crop_box_xyxy'][0]; y += context['crop_box_xyxy'][1]
                            qx = matrix[0,0]*x + matrix[0,1]*y + matrix[0,2]
                            qy = matrix[1,0]*x + matrix[1,1]*y + matrix[1,2]
                            qw = matrix[2,0]*x + matrix[2,1]*y + matrix[2,2]
                            assert abs(qw) > 1e-9
                            if l <= qx/qw <= r and t <= qy/qw <= b: hits[anchor['id']] += 1
                    signatures.append((len(part), tuple(sorted(hits.items()))))
                    if score >= .75 and not truncated and all(v > 0 for v in hits.values()): eligible += 1
                    if score >= .75 and hits['FAN_CPU'] > 0:
                        high_touch = True
                        socket_pixels.append(dict(mask=identity, component_pixels=len(part), socket_pixels=hits['FAN_CPU'], lead_pixels=hits['FAN_LEAD']))
                expected = [(c['pixel_count'], tuple(sorted(c['anchor_pixel_support'].items()))) for c in old['components']]
                assert sorted(signatures) == sorted(expected)
        assert eligible == len(view['eligible_native_components'])
        assert high_touch == view['high_score_mask_touches_socket']
        if case['id'] == 'reference':
            assert eligible == 1 and case['phenotype'] == 'mating_body_visible'
        else:
            exposed = case['phenotype'] == 'socket_contacts_exposed'
            expected = ('insufficient_evidence' if high_touch else 'visible_socket_attachment_change_supported') if exposed else ('same_visible_bundle_attachment_supported' if eligible == 1 else 'insufficient_evidence')
            assert comparisons[case['id']]['decision'] == expected
        checks.append(dict(id=case['id'], native_masks=masks, native_components=components,
            two_anchor_components=eligible, high_score_socket_component_hits=socket_pixels,
            decision='reference_supported' if case['id']=='reference' else comparisons[case['id']]['decision']))
    verify(protocol['pins']); assert source_pins() == protocol['mainline_pins']
    out = ROOT / 'artifacts/mendeley_typed_affine_bundle_native_audit_20261006'
    out.mkdir(exist_ok=False)
    save(out / 'report.json', dict(status='PASS', source_report_sha256=sha256(report_path),
        cases=checks, independent_python_flood_fill_and_scalar_projection=True,
        source_crops_and_manifest_bytes_verified=True, no_component_removed=True,
        shared_appearance_evidence_not_rerun=True, production_unchanged=True,
        newly_supported_cases=0, electrical_connections_confirmed=0, deployed=False))
    print(json.dumps(checks))

if __name__ == '__main__': main()
