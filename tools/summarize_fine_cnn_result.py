"""Report paired val outcomes and geometric review burden without promoting accuracy."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path('E:/PythonProject10');sys.path.insert(0,str(ROOT))
from tools.evaluate_mendeley_local_refinement import summarize
from tools.probe_cnn_candidate_rank import box_union_area


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('fresh path required')
    report=json.loads(args.report.read_text(encoding='utf-8'));assert report['split']=='val01' and report['status']=='complete'
    coarse=json.loads((ROOT/'output/mendeley_cnn_heat_evidence_20260929/original/report.json').read_text(encoding='utf-8'))
    x,y,r,b=coarse['bounds'];roi_area=(r-x)*(b-y)
    faults=[c for c in report['cases'] if c['targets']];normals=[c for c in report['cases'] if not c['targets']]
    per_image=[]
    for case in faults:
        old=summarize([case],'parents');new=summarize([case],'fine_selected')
        per_image.append({'image':case['image'],'old_overlap':old['target_boxes_with_overlap'],
              'new_overlap':new['target_boxes_with_overlap'],'old_precise':old['target_boxes_iou_ge_0_5'],
              'new_precise':new['target_boxes_iou_ge_0_5'],'target_count':old['target_boxes']})
    outcomes={'improved':sum(c['new_overlap']>c['old_overlap'] for c in per_image),
              'same':sum(c['new_overlap']==c['old_overlap'] for c in per_image),
              'regressed':sum(c['new_overlap']<c['old_overlap'] for c in per_image)}
    area={kind:{mode:float(np.mean([box_union_area(c[mode])/roi_area for c in group])) for mode in ('parents','fine_selected')}
          for kind,group in (('fault',faults),('normal',normals))}
    additional_regions={'fault':sum(len(c['hints']) for c in faults),'normal':sum(len(c['hints']) for c in normals)}
    macros={mode:float(np.mean([summarize([c],mode)['target_boxes_with_overlap']/len(c['targets']) for c in faults]))
            for mode in ('parents','fine_selected')}
    timing={key:float(np.mean([t[key] for t in report['timings'] if key in t])) for key in ('registration_seconds','fine_extraction_seconds','score_seconds')}
    result={'per_image':per_image,'overlap_outcomes':outcomes,'image_equal_overlap':macros,'union_roi_fraction':area,
            'additional_hint_regions':additional_regions,'normal_geometry_changes':sum(c['normal'] and c['geometry_changed'] for c in report['selection_changes']),
            'mean_stage_seconds':timing,'peak_observed_rss_gib':report['peak_observed_rss_bytes']/2**30,
            'bank_mib':report['bank_bytes']/2**20,'coarse_controls_count':len(report['manifest']),
            'coarse_controls_max_abs_drift':max(m['coarse_max_abs_error'] for m in report['manifest']),
            'all_registrations_reliable':all(m['alignment']['alignment_quality']['reliable'] for m in report['manifest']),
            'warning':'Geometric source-fragment comparisons and observed resource samples only; not field accuracy or a measured user-time reduction.'}
    with args.output.open('x',encoding='utf-8') as file:json.dump(result,file,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
