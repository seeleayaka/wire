"""Independently check completed fresh cases; never run or alter inference."""
import argparse
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'artifacts/fresh_four_examples_20261004'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def within(path, root):
    Path(path).resolve().relative_to(Path(root).resolve())
    return Path(path)


def reference_cache_records(value):
    if isinstance(value, dict):
        if 'reference_cache' in value:
            yield value['reference_cache']
        for child in value.values():
            yield from reference_cache_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from reference_cache_records(child)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    OUT=args.output.resolve()
    progress = read(OUT / 'progress.json')
    manifest = read(OUT / 'manifest.json')
    checked = []
    for row in progress['cases']:
        case = OUT / row.get('id',Path(row['image']).stem)
        assert row == read(case / 'result.json')
        output = within(row['output'], case / 'desktop_output')
        report = read(output / 'report.json')
        assert Path(report['reference']) == Path(row.get('reference',manifest['reference']))
        assert Path(report['inspection']) in map(Path, manifest['images'])
        assert Path(report['inspection']).name == row['image']
        task = read(within(row['task'], output))
        assert Path(task['inputs']['reference']) == Path(report['reference'])
        assert Path(task['inputs']['inspection']) == Path(report['inspection'])
        assert task['state'] == 'awaiting_human_review'
        assert not task['human_conclusions']
        assert row['human_confirmation'] is False
        assert row['fusion_cues'] == len(report['review_regions'])
        dino_caches = list(reference_cache_records(report))
        if row['sam_status'] == 'ok':
            assert dino_caches and all(value == 'miss' for value in dino_caches)
        ports = report.get('independent_port_rescue', {})
        assert row['ports_status'] == ports.get('status', 'not_run')
        assert row['port_reason'] == ports.get('fallback_reason')
        assert ports.get('automatic_fault_verdict', False) is False
        assert row['main_ports'] == len(ports.get('rescue_hints', []))
        assert row['supplementary_ports'] == len(ports.get('supplementary_hints', []))
        assert row['main_ports'] <= 5 and row['supplementary_ports'] <= 5
        sam = report.get('sam3_fusion', {})
        timings = {}
        if sam.get('status') == 'ok':
            for kind in ('reference', 'inspection'):
                evidence = sam[kind + '_sam3']
                assert evidence['cache_hit'] is False
                mask_dir = within(evidence['output_dir'], case)
                raw = read(mask_dir / 'report.json')
                expected_input = Path(row.get('reference',manifest['reference'])) if kind == 'reference' else output / 'aligned.jpg'
                assert Path(raw['input']) == expected_input
                assert str(Path(raw['checkpoint'])) in manifest['pins']
                assert raw['prompt'] == 'cable'
                assert raw['image_state_cache_reused'] is False
                assert raw['timings']['image_encoder_seconds'] > 0
                assert (mask_dir / 'mask_union.png').is_file()
                within(raw['image_state_cache'], case)
                timings[kind] = raw['timings']
            assert row['fresh_reference_SAM'] and row['fresh_inspection_SAM']
        else:
            assert report['decision'] == 'alignment_uncertain_manual_review'
        for key in ('comparison', 'desktop'):
            assert within(row[key], case).is_file()
        checked.append(dict(image=row['image'],sam_status=sam.get('status'),
                            decision=report['decision'],sam_timings=timings,
                            dino_reference_cache_misses=len(dino_caches),
                            fusion_cues=row['fusion_cues'],main_ports=row['main_ports'],
                            supplementary_ports=row['supplementary_ports']))
    if args.final:
        assert progress['status'] == 'complete' and len(checked) == len(manifest.get('cases',manifest['images']))
        assert read(OUT / 'report.json') == progress
        assert [r['image'] for r in checked] == [Path(p).name for p in manifest['images']]
        assert all(sha(path) == value for path, value in manifest['pins'].items())
    audit = dict(status='pass',scope='all_selected_final' if args.final else 'completed_cases_only',
                 runner_status=progress['status'],checked_cases=checked,
                 inference_started=False,accuracy_claim=False)
    target = OUT / ('freshness_audit.json' if args.final else 'partial_freshness_audit.json')
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
