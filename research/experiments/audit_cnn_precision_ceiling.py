"""Validation-only diagnostic ceilings, NOT deployable label-selected predictions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def reconstruct_groups(raw, tiled):
    pending = copy.deepcopy(raw); groups = []
    while pending:
        group = [pending.pop(0)]; changed = True
        while changed:
            changed = False; kept = []
            for candidate in pending:
                if any(tiled._iou(candidate, member)>=.25 or tiled._near_touching(candidate,member)
                       or tiled._contains(candidate,member) or tiled._contains(member,candidate) for member in group):
                    group.append(candidate); changed = True
                else: kept.append(candidate)
            pending = kept
        groups.append(group)
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('use fresh output')
    sys.path.insert(0,str(args.repo))
    from tools.merge_audit import setup
    from tools.evaluate_mendeley_local_refinement import summarize,iou,box_area
    _,tiled=setup(args.repo)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    frozen=load(args.repo/'output/mendeley_cnn_qualified_support_20260929/report.json')
    selected={c['image']:c for c in frozen['results']['cnn_calibrated_support']['cases']}
    records=[load(p) for directory in ('mendeley_merge_audit_20260928','mendeley_normal_evidence_20260928')
             for p in sorted((args.repo/'output'/directory).glob('*.json')) if p.name!='comparison.json']
    if len(records)!=30 or any(r['split']!='val01' for r in records): raise ValueError('requires fixed val01 only')
    source_sha=hashlib.sha256((args.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
    scenarios={name:[] for name in ('selected','all_merged','selected_group_raw','all_raw','raw_tile','raw_refinement','raw_whole')}
    observations=[]; diagnostics=[]
    for r in records:
        if r['source_sha256']!=source_sha: raise ValueError('source fingerprint changed')
        t=r['trace']; x,y,_,_=r['bounds']
        local_selected=[{**b,'left':b['left']-x,'right':b['right']-x,'top':b['top']-y,'bottom':b['bottom']-y}
                        for b in selected[r['image']]['candidates']]
        groups=reconstruct_groups(t['raw'],tiled)
        recovered=[tiled._merge_candidates(g)[0] for g in groups]
        merged=tiled._merge_candidates(copy.deepcopy(t['raw']))
        if sorted(recovered,key=key)!=sorted(merged,key=key): raise ValueError('group reconstruction mismatch')
        group_map={key(b):g for b,g in zip(recovered,groups)}
        if len(group_map)!=len(groups): raise ValueError('ambiguous identical groups')
        children=[b for parent in local_selected for b in group_map[key(parent)]]
        local={'selected':local_selected,'all_merged':merged,'selected_group_raw':children,'all_raw':t['raw']}
        for source in ('tile','refinement','whole'):
            local['raw_'+source]=[b for b in t['raw'] if b.get('source')==('whole_roi' if source=='whole' else source)]
        for parent in local_selected:
            members=group_map[key(parent)]
            ratio=[box_area(b)/box_area(parent) for b in members]
            observations.append({'image':r['image'],'normal':r['image'].startswith('normal_'),
                    'member_count':len(members),'min_member_parent_area_ratio':min(ratio),
                    'sources':sorted({b.get('source','unknown') for b in members})})
        for name,boxes in local.items():
            absolute=[{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in boxes]
            scenarios[name].append({'image':r['image'],'targets':r['targets'],'candidates':absolute})
        if r['targets']:
            best={name:[max((iou(b,target) for b in scenarios[name][-1]['candidates']),default=0)
                        for target in r['targets']] for name in scenarios}
            for index,target in enumerate(r['targets']):
                diagnostics.append({'image':r['image'],'target_index':index,'target_area':box_area(target),
                    'best_iou':{name:values[index] for name,values in best.items()}})
    def metrics(rows):
        faults=[r for r in rows if r['targets']]
        normals=[r for r in rows if not r['targets']]
        return {'faults':summarize(faults,'candidates'),'normal_candidate_count':sum(len(r['candidates']) for r in normals),
                'by_kind':{k:summarize([r for r in faults if r['image'].startswith(k+'_')],'candidates')
                           for k in ('damaged','disconnected','misrouted')}}
    result={name:metrics(rows) for name,rows in scenarios.items()}
    assert result['selected']['faults']==frozen['results']['cnn_calibrated_support']['overall']['faults']
    stages={'merge_lost_precise':sum(d['best_iou']['all_raw']>=.5 and d['best_iou']['all_merged']<.5 for d in diagnostics),
            'selection_lost_precise':sum(d['best_iou']['all_merged']>=.5 and d['best_iou']['selected']<.5 for d in diagnostics),
            'selected_group_has_hidden_precise':sum(d['best_iou']['selected_group_raw']>=.5 and d['best_iou']['selected']<.5 for d in diagnostics),
            'no_raw_precise':sum(d['best_iou']['all_raw']<.5 for d in diagnostics)}
    report={'split':'val01','source_sha256':source_sha,'groups_reproduced_exact':True,'results':result,
             'stage_counts':stages,'selected_group_observations':observations,'target_diagnostics':diagnostics,
             'warning':'All-pool and child scores are annotation-assisted geometric ceilings, unlimited candidates; not usable accuracy or an accepted algorithm. No test01 read, no production change.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as file:json.dump(report,file,indent=2)
    print(json.dumps({'results':result,'stage_counts':stages},indent=2))


if __name__=='__main__':main()
