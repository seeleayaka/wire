"""Check whether greedy IoU scoring hides feasible matches; never change gates.

Same class/IoU0.5 requirement; maximum-cardinality bipartite matching is only
an independent metric audit. Historical accepted/rejected scores stay intact.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,BASE,load,save,read_current_case,read_targets
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.port_tiling import box_iou
from audit_port_multiscale_acceptance import matches

OUT=ROOT/'artifacts/port_matching_cardinality_audit_20261003'


def maximum_matching(predictions,targets):
    edges={i:sorted([(box_iou(p['box_xyxy'],t['box']),j) for j,t in enumerate(targets)
                    if p['class_id']==t['class_id'] and box_iou(p['box_xyxy'],t['box'])>=.5],reverse=True)
           for i,p in enumerate(predictions)}
    target_owner={}
    def augment(pi,seen):
        for _,ti in edges[pi]:
            if ti in seen:continue
            seen.add(ti)
            if ti not in target_owner or augment(target_owner[ti],seen):
                target_owner[ti]=pi;return True
        return False
    for pi in range(len(predictions)):augment(pi,set())
    return target_owner


def main():
    if OUT.exists():raise FileExistsError('Preserve independent audit')
    frozen=resolution_runtime_fingerprint(REPO);pins={str(Path(__file__)):sha(Path(__file__))};OUT.mkdir()
    rows=[];sums={};deltas=[]
    for stage in ('train','inner','outer'):
        reportpath=BASE/stage/'report.json';pins[str(reportpath)]=sha(reportpath);total=dict(greedy_tp=0,maximum_tp=0,targets=0)
        for entry in load(reportpath)['cases']:
            teacher,record=read_current_case(stage,entry,pins)
            targets=read_targets(stage,entry['image'],teacher['predictions']['source_shape'],entry['label_sha256'],pins)
            predictions=record['trial']['all_predictions'];greedy=len(matches(predictions,targets)[0]);maximum=len(maximum_matching(predictions,targets))
            assert maximum>=greedy and greedy==entry['metrics']['trial']['tp']
            row=dict(stage=stage,image=entry['image'],greedy_tp=greedy,maximum_tp=maximum,targets=len(targets));rows.append(row)
            for k in total:total[k]+=row[k]
            if maximum!=greedy:deltas.append(row)
        sums[stage]=total
    assert len(rows)==270 and {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
    save(OUT/'report.json',dict(status='complete',summary=sums,differences=deltas,cases=rows,pins=pins,
         scoring_rules_unchanged=True,source_cached=True,field_accuracy=False))
    print(str(dict(summary=sums,differences=len(deltas))),flush=True)


if __name__=='__main__':main()
