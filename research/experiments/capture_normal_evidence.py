"""Capture all validation normals for isolated gate stress testing."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('E:/PythonProject10'))
    args = parser.parse_args()
    repo = args.repo
    sys.path.insert(0, str(repo))
    from tools.merge_audit import setup, save
    impl, tiled = setup(repo)
    from evaluate_mendeley_balanced import FULL_REVIEW_ROI, read_image
    import assembly_auto_review_robust_v3 as perspective
    original_merge, original_review = tiled._merge_candidates, impl.review_components
    raw, traces = [], []
    def merge_hook(rows):
        raw.append(copy.deepcopy(rows))
        return original_merge(rows)
    def review_hook(reference, aligned, valid, **kwargs):
        before = len(raw)
        result = original_review(reference, aligned, valid, **kwargs)
        assert len(raw) == before + 1
        traces.append({'raw':raw[-1], 'height':reference.shape[0], 'width':reference.shape[1],
                       'local_candidates':copy.deepcopy(result[2]), 'metadata':copy.deepcopy(result[1])})
        return result
    tiled._merge_candidates, impl.review_components = merge_hook, review_hook
    tiled.LARGE_ROI_MAX_CANDIDATES = 6
    dataset = repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    reference = read_image(dataset/'images/train01/normal_073.JPG')
    bounds = impl.adaptive.local_review.motion.perspective.auto.base.pixels(reference, FULL_REVIEW_ROI[0])
    paths = sorted((dataset/'images/val01').glob('normal_*.JPG'))
    assert len(paths) == 15
    try:
        for n,path in enumerate(paths,1):
            raw.clear(); traces.clear()
            aligned, _ = perspective.automatic_homography(reference, read_image(path))
            if aligned is None:
                raise RuntimeError(f'alignment failed: {path.name}')
            _, _, candidates = impl.dino_fused_regions(reference, aligned, FULL_REVIEW_ROI)
            assert len(traces) == 1
            save(args.output/(path.stem+'.json'), {'image':path.name,'split':'val01','bounds':bounds,
                'trace':traces[0],'candidates':candidates,'targets':[],
                'source_sha256':hashlib.sha256((repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()})
            print(f'{n}/15 {path.name}: raw={len(raw[0])}, final={len(candidates)}',flush=True)
    finally:
        tiled._merge_candidates, impl.review_components = original_merge, original_review

if __name__ == '__main__':
    main()
