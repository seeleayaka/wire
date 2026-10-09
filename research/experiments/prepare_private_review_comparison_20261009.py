"""Prepare a bounded comparison and unapproved local previews. No API client."""
import json
from pathlib import Path
import launch_llm_recheck_window_20261008
import private_region_review as private
import llm_review_priority as priority
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase,QFont
from probe_llm_recheck_planner_20261008 import ROOT,CASES

OUT=ROOT/'artifacts/private_review_comparison_20261009'
def pending(source):
    report=json.loads(source.read_text(encoding='utf-8'));f=report['sam3_fusion']
    return {'report':report,'output':source.parent,'candidates':report['review_regions'],
            'reference_mask':Path(f['reference_sam3']['output_dir'])/'mask_union.png',
            'inspection_mask':Path(f['inspection_sam3']['output_dir'])/'mask_union.png'}
def main():
    app=QApplication([]);fid=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    app.setFont(QFont(QFontDatabase.applicationFontFamilies(fid)[0],10));OUT.mkdir(exist_ok=True)
    jobs=[]
    for name,source in CASES.items():
        p=pending(source);prepared=private.prepare(p)
        for r in prepared['regions']:
            for side in range(2):
                im=r['images'][side]
                path=OUT/f'{name}_{r["candidate_id"]}_{side}_LOCAL_RAW.png'
                assert im.save(str(path),'PNG')
        evidence=priority.local_evidence(p['report'],p['reference_mask'],p['inspection_mask'])
        jobs.append({'case':name,'source_report_sha256':private.sha(source),
                     'source_binding':prepared['source_binding'],'candidate_count':len(evidence),
                     'local_evidence':evidence,'external_approval':False})
    plan={'design':'paired_two_cases_same_candidates_same_priority_gate',
          'requested_model':'unchanged existing configuration','maximum_future_requests':4,
          'per_case':['binary_masks','redacted_local_photos_with_local_masks'],
          'timeout_seconds':45,'automatic_retries':0,'new_requests_performed':0,
          'primary_measures':['valid_schema','concrete_supported_observations','unsupported_claims',
                              'evidence_gated_priority_changes','elapsed_seconds'],
          'not_measured':['field_accuracy','electrical_continuity_accuracy','recall_gain'],
          'sam_rerun':False,'production_modified':False,'cases':jobs}
    (OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(jobs),'candidate_count':sum(x['candidate_count'] for x in jobs),
                      'new_api_calls':0,'eligibility':[x['local_evidence'] for x in jobs]},ensure_ascii=False))
if __name__=='__main__':main()
