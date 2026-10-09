"""Independent original-label/IoU scoring and actual replacement evidence audit."""
import json
import math
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,REFERENCE_SHA
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA
from consensus_rank import select
from replacement_voters import check_prefix
from inspection_agent.optional_port_crop_review import CONFIG
from inspection_agent.port_tiling import tile_windows,merge_tiled_ports
from exact_paired_score_cache import binding as semantic_binding,reuse as semantic_reuse

SOURCE=ROOT/'artifacts/allport480_source_20261005'
OUT=ROOT/'artifacts/allport480_source_audit_20261005'


def audit_views(evidence,roles,source_sha,shape):
    """Check actual distinct recipes; an empty placeholder is not a view."""
    if set(roles)!={'teacher','student','feature'} or len(set(roles.values()))!=3:
        raise ValueError('exactly three distinct roles required')
    height,width=shape
    grids={'global':[[0,0,width,height]],
           'native':[list(w) for w in tile_windows(width,height,1280,960)],
           'fine':[list(w) for w in tile_windows(width,height,960,720)]}
    actual=set()
    for view in evidence:
        digest=view['weight_sha256'];pred=view['predictions']
        if digest not in roles.values() or view['source_sha256']!=source_sha or pred['source_shape']!=list(shape):
            raise ValueError('view identity/frame mismatch')
        recipes=[key for key,windows in grids.items() if pred.get('windows')==windows]
        if len(recipes)!=1 or (digest,recipes[0]) in actual:
            raise ValueError('missing or duplicate actual view recipe')
        recipe=recipes[0];actual.add((digest,recipe))
        if recipe!='global' and pred['merged_predictions']!=merge_tiled_ports(pred['edge_kept_predictions'],CONFIG['cross_tile_nms_iou']):
            raise ValueError('tiled view merge replay mismatch')
        for row in pred['merged_predictions']:
            box=row['box_xyxy'];confidence=row['confidence'];cls=row['class_id']
            if (len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box)
                or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height):
                raise ValueError('invalid source-pixel detector geometry')
            if (type(cls) is not int or cls not in (0,1) or isinstance(confidence,bool)
                or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1):
                raise ValueError('invalid actual detector class/confidence')
    if actual!={(digest,recipe) for digest in roles.values() for recipe in grids}:
        raise ValueError('each actual role requires global/native/fine evidence')
    return len(actual)


def audit_probabilities(proposals,probabilities):
    if len(proposals)!=len(probabilities):raise ValueError('semantic proposal count mismatch')
    for row in probabilities:
        if (len(row)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in row)
            or abs(sum(row)-1)>1e-5):raise ValueError('invalid semantic probability vector')


def audit_semantic_reuse(case,report,required=False):
    record=case.get('semantic_score_reuse')
    if record is None:
        if required and case['proposals']:raise ValueError('missing semantic cache provenance')
        return 0,len(case['proposals'])
    parentpath=Path(record['source_case_path']);parentreportpath=parentpath.parent.parent/'report.json'
    if (report['pins'].get(str(parentpath))!=record['source_case_sha256'] or sha(parentpath)!=record['source_case_sha256']
        or report['pins'].get(str(parentreportpath))!=sha(parentreportpath)):
        raise ValueError('semantic parent outputs/report not bound')
    parent=load(parentpath);parentreport=load(parentreportpath)
    if parent['head_sha256']!=HEAD_SHA or parent['image']!=case['image']:raise ValueError('cached semantic identity changed')
    referencepath=DATA/'images/train01/normal_073.JPG'
    old_binding=semantic_binding(parent['source_sha256'],parentreport['pins'][str(referencepath)],parentreport['runtime'],parent['alignment'],[2736,3648])
    new_binding=semantic_binding(case['source_sha256'],REFERENCE_SHA,report['runtime'],case['alignment'],[2736,3648])
    scores,missing,reused=semantic_reuse(case['proposals'],parent['proposals'],parent['probabilities'],old_binding,new_binding)
    if (record['old_input_binding']!=old_binding or record['new_input_binding']!=new_binding
        or record['fresh_indices']!=missing or record['reused_count']!=reused or record['same_classifier_not_an_extra_vote'] is not True):
        raise ValueError('semantic cache index/provenance replay mismatch')
    for index,value in enumerate(scores):
        if value is not None and value!=case['probabilities'][index]:raise ValueError('exact cached semantic score changed')
    return reused,len(missing)


def iou(a,b):
    width=max(0,min(a[2],b[2])-max(a[0],b[0]));height=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=width*height
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def score(predictions,targets):
    pairs=[]
    for pi,p in enumerate(predictions):
        for ti,t in enumerate(targets):
            if p['class_id']==t['class_id']:pairs.append((iou(p['box_xyxy'],t['box']),pi,ti))
    used_p=set();used_t=set()
    for value,pi,ti in sorted(pairs,reverse=True):
        if value>=.5 and pi not in used_p and ti not in used_t:used_p.add(pi);used_t.add(ti)
    return dict(tp=len(used_t),unmatched=len(predictions)-len(used_p),fn=len(targets)-len(used_t),
                predictions=len(predictions),targets=len(targets)),used_t


def main(source=SOURCE,out=OUT):
    global SOURCE,OUT
    SOURCE=Path(source).resolve();OUT=Path(out).resolve()
    if OUT.exists():raise FileExistsError('preserve independent source audit')
    protocol_sha=sha(SOURCE/'protocol.json');report_sha=sha(SOURCE/'report.json')
    protocol=load(SOURCE/'protocol.json');report=load(SOURCE/'report.json')
    if report['status'] not in ('rejected','source_pass_requires_independent_replay_and_fresh_holdouts'):
        raise ValueError('final complete source inference is required')
    if len(report['cases'])!=192 or any(sha(p)!=v for p,v in report['pins'].items()):raise ValueError('source provenance drift/incomplete')
    if report['runtime']!=native_pose_runtime_fingerprint(REPO):raise ValueError('mainline fingerprint drift')
    roles=protocol['roles'];allowed=set(roles.values())
    if len(allowed)!=3:raise ValueError('wrong role population')
    names=sorted(r['image'] for r in report['cases'])
    if len(set(names))!=192 or names!=sorted(protocol['train_sources']):raise ValueError('duplicate/missing source membership')
    expected_files={Path(name).stem+'_predictions.json' for name in names}
    if {p.name for p in (SOURCE/'train').glob('*_predictions.json')}!=expected_files:
        raise ValueError('source output inventory mismatch')
    from protected_allport_baseline import check_accepted,check_totals
    recovery=report.get('aggregation_recovery');acceptedrows=[]
    rows=[];views=0;proposals=0;new_cues=0;skips={};case_pins={};reused_scores=0;fresh_scores=0
    for item in report['cases']:
        name=item['image'];p=SOURCE/'train'/(Path(name).stem+'_predictions.json');case_pins[str(p)]=sha(p);case=load(p)
        if case['image']!=name or case['roles']!=roles or case['head_sha256']!=HEAD_SHA:raise ValueError('source role/head mismatch')
        original=case['original'];current=case['current'];trial=case['trial'];check_prefix(original,current,trial)
        acceptedpath=ROOT/'artifacts/paired_pose_native_three_20261004/full_train'/(Path(name).stem+'_predictions.json')
        if report['pins'].get(str(acceptedpath))!=sha(acceptedpath):raise ValueError('accepted295 output is not bound')
        accepted=load(acceptedpath)['trial']
        if recovery:check_accepted(original,accepted,current,trial)
        elif original!=accepted:raise ValueError('wrong original baseline generation')
        source=DATA/'images/train01'/name
        if sha(source)!=case['source_sha256'] or report['pins'][str(source)]!=case['source_sha256']:raise ValueError('source pixels drift')
        if case['eligible']:
            evidence=case['new_voter_views']
            audit_views(evidence,roles,case['source_sha256'],[2736,3648])
            if case['alignment']['alignment_quality']['reliable'] is not True:raise ValueError('bad reference alignment')
            pool=[(v['weight_sha256'],r) for v in evidence for r in v['predictions']['merged_predictions']
                  if r['confidence']>.05 and 16<=r['box_xyxy'][0]<r['box_xyxy'][2]<=3632 and 16<=r['box_xyxy'][1]<r['box_xyxy'][3]<=2720]
            for candidate in case['proposals']:
                best={}
                for digest,row in pool:
                    if row['class_id']!=candidate['class_id']:continue
                    value=iou(row['box_xyxy'],candidate['box_xyxy'])
                    if value>=.5:best[digest]=max(best.get(digest,0),value)
                if sorted(best)!=sorted(allowed) or sorted(best)!=candidate['semantic_model_vote_sha256']:raise ValueError('not three real replacement votes')
                if best!=candidate['localization_voter_best_IoU']:raise ValueError('localization rank replay mismatch')
            views+=len(evidence)
        else:
            if case['proposals'] or case['probabilities'] or case['new_voter_views']:raise ValueError('abstention acquired evidence')
            reason=case['skip_reason'];skips[reason]=skips.get(reason,0)+1
        audit_probabilities(case['proposals'],case['probabilities'])
        reused,fresh=audit_semantic_reuse(case,report,protocol.get('exact_semantic_cache_only',False))
        reused_scores+=reused;fresh_scores+=fresh
        replay=select(current,case['proposals'],case['probabilities'],HEAD_SHA)
        if replay!=trial:raise ValueError('source selector replay mismatch')
        proposals+=len(case['proposals']);new_cues+=len(trial['all_predictions'])-len(current['all_predictions'])
        for cue in trial['all_predictions'][len(current['all_predictions']):]:
            if cue['paired_semantic_probability']<.98 or cue['paired_head_sha256']!=HEAD_SHA or cue['automatic_fault_verdict'] is not False:
                raise ValueError('new cue semantic/verdict contract mismatch')
        # Independent GT decoding happens only after the saved inference is verified.
        label=DATA/'labels/train01'/(Path(name).stem+'.txt')
        if sha(label)!=report['pins'][str(label)]:raise ValueError('original GT drift')
        targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            values=list(map(float,line.split()));cls,cx,cy,bw,bh=values
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*3648,(cy-bh/2)*2736,(cx+bw/2)*3648,(cy+bh/2)*2736]))
        metrics={};hits={}
        for version in ('original','current','trial'):metrics[version],hits[version]=score(case[version]['all_predictions'],targets)
        expected=dict(image=name,**metrics,gained=sorted(hits['trial']-hits['current']),lost=sorted(hits['current']-hits['trial']),
                      lost_original=sorted(hits['original']-hits['trial']),skip_reason=case['skip_reason'])
        if expected!=item:raise ValueError('independent source metric mismatch')
        acceptedmetrics,acceptedhits=score(accepted['all_predictions'],targets)
        acceptedrows.append(dict(image=name,metrics=acceptedmetrics,lost=sorted(acceptedhits-hits['trial']),path=str(acceptedpath),sha256=sha(acceptedpath)))
        rows.append(expected)
    totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('original','current','trial')}
    acceptedtotal={k:sum(r['metrics'][k] for r in acceptedrows) for k in ('tp','unmatched','fn','predictions','targets')}
    check_totals(dict(**totals,accepted=acceptedtotal),legacy=bool(recovery))
    if any(r['lost'] for r in acceptedrows):raise ValueError('accepted295 real hits lost')
    if recovery and (recovery['accepted_summary']!=acceptedtotal or recovery['accepted_cases']!=acceptedrows
                     or recovery['saved_source_case_sha256']!=case_pins):raise ValueError('recovery accepted/output binding mismatch')
    normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
    qualifies=totals['trial']['tp']>298 and totals['trial']['unmatched']<=4 and normal==0 and not any(r['lost'] or r['lost_original'] for r in rows)
    if totals!=report['summary'] or qualifies!=report['qualifies'] or normal!=report['normal_cues']:raise ValueError('independent final gate mismatch')
    if protocol.get('exact_semantic_cache_only') and (report['reused_semantic_scores']!=reused_scores or report['fresh_semantic_scores']!=fresh_scores):
        raise ValueError('semantic cache aggregate count mismatch')
    if (sha(SOURCE/'report.json')!=report_sha or sha(SOURCE/'protocol.json')!=protocol_sha
        or any(sha(p)!=d for p,d in case_pins.items()) or any(sha(p)!=d for p,d in report['pins'].items())
        or native_pose_runtime_fingerprint(REPO)!=report['runtime']):raise ValueError('evidence changed during independent audit')
    OUT.mkdir();final=dict(status='pass',candidate_source_qualifies=qualifies,summary=totals,normal_cues=normal,
        sources=192,actual_voter_views=views,proposal_replays=proposals,new_cues=new_cues,abstentions=skips,
        independent_original_GT_and_matching=True,source_report_sha256=report_sha,source_protocol_sha256=protocol_sha,
        source_case_sha256=case_pins,auditor_sha256=sha(__file__),actual_distinct_view_recipes=True,
        reused_semantic_scores=reused_scores,fresh_semantic_scores=fresh_scores,cache_replay_exact=True,
        accepted_mainline_source_summary=acceptedtotal,accepted_mainline_prefix_exact=True,
        aggregation_recovered_posthoc=bool(recovery),original_worker_succeeded=not bool(recovery),
        selector_replay_exact=True,no_model_inference=True,no_heldout_reads=True,no_deployment=True,field_accuracy=None,
        warning='Reused TRAIN192 source development audit, not cross-cabinet electrical-fault/continuity accuracy. Normal abstentions are not normal recognition successes.')
    save(OUT/'report.json',final);print(json.dumps(final))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();main(args.source,args.output)
