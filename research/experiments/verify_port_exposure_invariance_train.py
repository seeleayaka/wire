"""Independent pixel/statistic replay, not a recognition acceptance test."""
import sys
from pathlib import Path
from collections import Counter
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, read_image
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint

SOURCE = ROOT / 'artifacts/port_exposure_invariance_train_20261004'
OUT = ROOT / 'artifacts/port_exposure_invariance_replay_20261004'


def independently_reconstruct(image, expected, mask):
    import numpy as np
    sampled = image[::32, ::32].astype(float)
    target = expected[::32, ::32].astype(float)
    registered = mask[::32, ::32] > .99
    gains, mads, counts = [], [], []
    for channel in range(3):
        a, b = sampled[:, :, channel], target[:, :, channel]
        keep = registered & (a >= 16) & (a <= 239) & (b >= 16) & (b <= 239)
        counts.append(int(keep.sum()))
        if counts[-1] < 512:
            return image.copy(), dict(status='abstained', reason='insufficient_registered_unsaturated_samples', sample_counts=counts)
        differences = np.log(b[keep]) - np.log(a[keep])
        center = float(np.median(differences))
        gains.append(float(np.exp(center)))
        mads.append(float(np.median(np.abs(differences - center))))
    info = dict(channel_gain=gains, log_ratio_MAD=mads, sample_counts=counts,
                method='uniform32_registered_pair_log_median', no_GT_or_metadata=True)
    if min(gains) < .7 or max(gains) > 1.4:
        return image.copy(), dict(status='abstained', reason='gain_outside_fixed_safe_range', **info)
    if max(mads) > .12:
        return image.copy(), dict(status='abstained', reason='nonuniform_or_unreliable_photometric_pair', **info)
    if max(abs(v - 1) for v in gains) <= .025:
        return image.copy(), dict(status='identity', reason='already_near_matched_exposure', **info)
    result = image.astype(np.float32)
    for c, gain in enumerate(gains):
        result[:, :, c] *= np.float32(gain)
    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    # No reference image pixels are copied into the transformed image.
    assert (result[image == 0] == 0).all()
    return result, dict(status='compensated', detector_branch_only=True,
                        never_replace_visual_report_pixels=True, **info)


def main():
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    if OUT.exists():
        raise FileExistsError('Preserve independent exposure replay')
    report = load(SOURCE / 'report.json')
    assert report['status'] == 'complete'
    assert report['no_label_files_read'] and report['no_fitting'] and report['no_detector_inference']
    pins = report['pins']
    assert all(sha(Path(p)) == value for p, value in pins.items())
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    assert not any('/labels/' in p.replace('\\', '/') for p in pins)
    expected_names = sorted(load(ROOT / 'artifacts/port_training_multiscale_20261002/protocol.json')['train_sources'])
    assert len(expected_names) == 192 and [row['image'] for row in report['cases']] == expected_names
    reference = read_image(DATA / 'images/train01/normal_073.JPG')
    counts, relative = Counter(), []
    for i, row in enumerate(report['cases']):
        name = row['image']
        record = load(ROOT / 'artifacts/paired_port_semantics_20261003/features_train' / (Path(name).stem + '_source.json'))
        assert record['source_sha256'] == sha(DATA / 'images/train01' / name)
        alignment = record['alignment']
        if not alignment.get('alignment_quality', {}).get('reliable'):
            assert row['status'] == 'abstained_registration'
            counts['unreliable_registration'] += 1
            continue
        original = read_image(DATA / 'images/train01' / name)
        h, w = original.shape[:2]
        inverse = np.linalg.inv(np.array(alignment['source_to_reference_homography'], dtype=float))
        expected = cv2.warpPerspective(reference, inverse, (w, h), flags=cv2.INTER_LINEAR)
        registered = cv2.warpPerspective(np.full(reference.shape[:2], 255, dtype=np.uint8), inverse, (w, h), flags=cv2.INTER_NEAREST)
        mask = cv2.erode(registered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        dark = np.clip(np.rint(original.astype(np.float32) * .85), 0, 255).astype(np.uint8)
        a, first = independently_reconstruct(original, expected, mask)
        b, second = independently_reconstruct(dark, expected, mask)
        assert first == row['original_policy'] and second == row['dark_policy'], name
        valid = mask[::32, ::32] > .99
        before = np.abs(dark[::32, ::32].astype(float) - original[::32, ::32]).mean(axis=2)[valid].mean()
        after = np.abs(a[::32, ::32].astype(float) - b[::32, ::32]).mean(axis=2)[valid].mean()
        assert abs(before - row['original_exposure_difference_MAE']) < 1e-10
        assert abs(after - row['normalized_pair_difference_MAE']) < 1e-10
        ratio = float(after / before) if before else None
        assert ratio == row['relative_difference']
        status = 'both_usable' if first['status'] != 'abstained' and second['status'] != 'abstained' else 'at_least_one_abstained'
        assert row['status'] == status and row['not_recognition_metrics']
        counts[status] += 1
        if status == 'both_usable':
            relative.append(ratio)
        if (i + 1) % 32 == 0:
            print(dict(completed=i + 1, total=192), flush=True)
    assert dict(counts) == report['counts']
    assert float(np.median(relative)) == report['median_relative_difference']
    assert all(sha(Path(p)) == value for p, value in pins.items())
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    OUT.mkdir()
    result = dict(status='pass', train_sources=192, counts=dict(counts),
                  median_relative_difference=float(np.median(relative)),
                  report_sha256=sha(SOURCE / 'report.json'), auditor_sha256=sha(Path(__file__)),
                  independent_warp_and_estimator=True, no_label_files_read=True,
                  no_fitting=True, no_detector_inference=True, no_deployment=True,
                  not_recognition_accuracy=True)
    save(OUT / 'report.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
