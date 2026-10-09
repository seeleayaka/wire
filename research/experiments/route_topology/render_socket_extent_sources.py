"""Render completed source controls without changing pinned evidence or decisions."""
from pathlib import Path
from analyze_mendeley_scope import render_all
from run_paired_evidence import load_run
from run_review import read, save
from run_prompt_contrast import verify
from core import sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    evidence = ROOT / 'artifacts/mendeley_socket_extent_source_controls_20261006'
    protocol = read(evidence / 'protocol.json')
    inference = read(evidence / 'inference_report.json')
    if inference['status'] != 'complete':
        raise ValueError('source inference still incomplete')
    if inference['protocol_sha256'] != sha256(evidence / 'protocol.json'):
        raise ValueError('source protocol drift')
    verify(protocol['pins'])
    output = ROOT / 'artifacts/mendeley_socket_extent_source_visuals_20261006'
    output.mkdir(exist_ok=False)
    sheets = []
    for case in read(evidence / 'preparation_report.json')['cases']:
        if not case['sam_inference_requested']:
            continue
        destination = output / case['id']
        destination.mkdir()
        for recipe in protocol['recipes']:
            origin, records = load_run(evidence / case['id'] / recipe)
            render_all(origin, records, destination / recipe)
            sheets.append({'case': case['id'], 'recipe': recipe,
                           'masks': len(records),
                           'sheet': str(destination / recipe / 'all_masks.png')})
    verify(protocol['pins'])
    save(output / 'render_report.json', {'status': 'rendered', 'sheets': sheets,
         'actual_visual_review': 'pending', 'not_an_acceptance_report': True})
    print({'rendered_sheets': len(sheets)})


if __name__ == '__main__':
    main()
