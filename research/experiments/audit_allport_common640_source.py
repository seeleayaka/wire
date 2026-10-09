"""Independent original-GT/common640 evidence replay. No models/fitting/deployment."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint, HEAD_SHA
from audit_allport480_source import audit_probabilities, audit_semantic_reuse, score, iou
from common_scale_evidence import audit_common_views, check_common_prefixes, common_source_qualifies
from consensus_rank import select
from raw_pose_consensus import proposals
from fine_voter_quality import attach
from inspection_agent.paired_port_features import expected_in_source, valid_boxes
from inspection_agent.optional_port_crop_review import read_image, REFERENCE_SHA

SOURCE = ROOT/'artifacts/allport_common640_source_20261005'
OUT = ROOT/'artifacts/allport_common640_source_audit_20261005'
PARENT = ROOT/'artifacts/allport480_teacher_source_20261005'


def audit_new_proposal_votes(case, roles):
    pool = [(v['weight_sha256'], r) for v in case['new_voter_views']
            for r in v['predictions']['merged_predictions']
            if r['confidence'] > .05 and 16 <= r['box_xyxy'][0] < r['box_xyxy'][2] <= 3632
            and 16 <= r['box_xyxy'][1] < r['box_xyxy'][3] <= 2720]
    for proposal in case['proposals']:
        best = {}
        for digest, row in pool:
            if row['class_id'] != proposal['class_id']: continue
            overlap = iou(row['box_xyxy'], proposal['box_xyxy'])
            if overlap >= .5: best[digest] = max(best.get(digest, 0), overlap)
        if (set(best) != set(roles.values()) or sorted(best) != proposal['semantic_model_vote_sha256']
                or best != proposal['localization_voter_best_IoU']):
            raise ValueError('proposal lacks three actual checkpoint localization votes')


def audit_proposal_generation(case, reference):
    """Reproduce GT-free seed poses and frozen valid-reference context, not just votes."""
    frame = dict(source_sha256=case['source_sha256'], predictions={'source_shape': [2736, 3648]})
    _, valid = expected_in_source(reference, case['alignment']['source_to_reference_homography'], [2736, 3648])
    generated = proposals(frame, case['new_voter_views'], case['current'])
    generated = [generated[i] for i in valid_boxes([r['box_xyxy'] for r in generated], valid)]
    generated = attach(generated, case['new_voter_views'], case['source_sha256'], [2736, 3648])
    if generated != case['proposals']:
        raise ValueError('saved proposals differ from GT-free seed/context replay')


def main():
    if OUT.exists(): raise FileExistsError('preserve independent common640 source audit')
    reportpath, protocolpath = SOURCE/'report.json', SOURCE/'protocol.json'
    reportsha, protocolsha = sha(reportpath), sha(protocolpath)
    report, protocol = load(reportpath), load(protocolpath)
    if report['status'] not in ('rejected', 'source_pass_requires_independent_replay_and_fresh_holdouts'):
        raise ValueError('complete final source report required; partial/live output is not acceptance')
    if (report['pins'].get(str(Path(__file__))) != sha(__file__)
            or any(sha(p) != d for p, d in report['pins'].items())
            or native_pose_runtime_fingerprint(REPO) != report['runtime']):
        raise ValueError('frozen auditor/runtime/input drift')
    parent = load(PARENT/'report.json')
    parentauditpath = ROOT/'artifacts/allport480_teacher_source_audit_20261005/report.json'
    parentaudit = load(parentauditpath)
    if (report['pins'].get(str(PARENT/'report.json')) != sha(PARENT/'report.json')
            or parentaudit['source_report_sha256'] != sha(PARENT/'report.json')
            or parentaudit['status'] != 'pass' or not parentaudit['candidate_source_qualifies']
            or any(sha(p) != d for p, d in parent['pins'].items())):
        raise ValueError('audited parent source300 not bound')
    roles = protocol['roles']; names = [r['image'] for r in report['cases']]
    if (len(names) != 192 or len(set(names)) != 192 or sorted(names) != sorted(protocol['train_sources'])
            or sorted(names) != sorted(r['image'] for r in parent['cases'])):
        raise ValueError('exact complete unique TRAIN192 cohort required')
    if {p.name for p in (SOURCE/'train').glob('*_predictions.json')} != {Path(n).stem+'_predictions.json' for n in names}:
        raise ValueError('saved case inventory mismatch')
    referencepath = DATA/'images/train01/normal_073.JPG'
    if sha(referencepath) != REFERENCE_SHA: raise ValueError('reference pixels drift')
    reference = read_image(referencepath)
    rows = []; case_pins = {}; actual_views = proposal_count = reused_scores = fresh_scores = 0
    for item in report['cases']:
        name = item['image']; stem = Path(name).stem
        path = SOURCE/'train'/(stem+'_predictions.json'); case_pins[str(path)] = sha(path); case = load(path)
        parentpath = PARENT/'train'/(stem+'_predictions.json'); previous = load(parentpath)
        if report['pins'].get(str(parentpath)) != parentaudit['source_case_sha256'][str(parentpath)]:
            raise ValueError('audited parent case identity mismatch')
        if sha(parentpath) != report['pins'][str(parentpath)]: raise ValueError('parent output drift')
        sourcepath = DATA/'images/train01'/name
        if (case['image'] != name or case['roles'] != roles or case['head_sha256'] != HEAD_SHA
                or sha(sourcepath) != case['source_sha256'] or report['pins'][str(sourcepath)] != case['source_sha256']):
            raise ValueError('actual image/checkpoint/head identity drift')
        if (case['original'] != previous['original'] or case['research'] != previous['current']
                or case['current'] != previous['trial'] or case['alignment'] != previous['alignment']):
            raise ValueError('source generations/reference geometry changed')
        check_common_prefixes(case['original'], case['research'], case['current'], case['trial'])
        expected_eligible = previous['eligible'] and len(case['current']['all_predictions'])-len(case['current']['primary']) < 5
        if case['eligible'] != expected_eligible: raise ValueError('GT-free source eligibility changed')
        if case['eligible']:
            if case['alignment']['alignment_quality']['reliable'] is not True: raise ValueError('unreliable source acquired new views')
            actual_views += audit_common_views(case['new_voter_views'], roles, case['source_sha256'], [2736, 3648])
            # The original9 views MUST be the actual audited parent, not substituted.
            if case['new_voter_views'][:9] != previous['new_voter_views']:
                raise ValueError('reused actual parent view prefix changed')
            audit_proposal_generation(case, reference)
            audit_new_proposal_votes(case, roles)
        elif case['new_voter_views'] or case['proposals'] or case['probabilities']:
            raise ValueError('abstention acquired evidence')
        audit_probabilities(case['proposals'], case['probabilities'])
        reused, fresh = audit_semantic_reuse(case, report, required=True)
        reused_scores += reused; fresh_scores += fresh; proposal_count += len(case['proposals'])
        if select(case['current'], case['proposals'], case['probabilities'], HEAD_SHA) != case['trial']:
            raise ValueError('unchanged selector replay mismatch')
        for cue in case['trial']['all_predictions'][len(case['current']['all_predictions']):]:
            if (cue['paired_semantic_probability'] < .98 or cue['paired_head_sha256'] != HEAD_SHA
                    or cue['automatic_fault_verdict'] is not False):
                raise ValueError('new cue semantic/verdict contract mismatch')
        # Only saved predictions are scored. Decode the original labels separately.
        labelpath = DATA/'labels/train01'/(stem+'.txt')
        if sha(labelpath) != report['pins'][str(labelpath)]: raise ValueError('original GT drift')
        targets = []
        for line in labelpath.read_text(encoding='utf-8').splitlines():
            cls, x, y, w, h = map(float, line.split())
            if cls in (3, 4):
                targets.append(dict(class_id=int(cls)-3, box=[(x-w/2)*3648, (y-h/2)*2736, (x+w/2)*3648, (y+h/2)*2736]))
        metrics = {}; hits = {}
        for version in ('original', 'research', 'current', 'trial'):
            metrics[version], hits[version] = score(case[version]['all_predictions'], targets)
        row = dict(image=name, **metrics, gained=sorted(hits['trial']-hits['current']),
                   lost=sorted(hits['current']-hits['trial']), lost_original=sorted(hits['original']-hits['trial']),
                   lost_research=sorted(hits['research']-hits['trial']), skip_reason=case['skip_reason'])
        if row != item: raise ValueError('independent original-GT/source metric mismatch')
        rows.append(row)
    totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')}
              for v in ('original', 'research', 'current', 'trial')}
    normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
    qualifies = common_source_qualifies(totals, normal, rows)
    if (totals != report['summary'] or qualifies != report['qualifies'] or normal != report['normal_cues']
            or reused_scores != report['reused_semantic_scores'] or fresh_scores != report['fresh_semantic_scores']):
        raise ValueError('independent full-source gate/score reuse aggregate mismatch')
    if (sha(reportpath) != reportsha or sha(protocolpath) != protocolsha or any(sha(p) != d for p, d in case_pins.items())
            or any(sha(p) != d for p, d in report['pins'].items()) or native_pose_runtime_fingerprint(REPO) != report['runtime']):
        raise ValueError('frozen evidence changed during replay')
    OUT.mkdir()
    save(OUT/'report.json', dict(status='pass', candidate_source_qualifies=qualifies, summary=totals, normal_cues=normal,
         source_report_sha256=reportsha, source_protocol_sha256=protocolsha, source_case_sha256=case_pins,
         auditor_sha256=sha(__file__), actual_view_replays=actual_views, proposal_replays=proposal_count,
         reused_semantic_scores=reused_scores, fresh_semantic_scores=fresh_scores, original_GT_independent=True,
         protected_prefixes_exact=[295, 298, 300], no_model_inference=True, no_deployment=True, field_accuracy=None,
         warning='Repeated TRAIN development data; not cross-cabinet accuracy, normal recognition or electrical continuity.'))
    print(dict(status='pass', candidate_source_qualifies=qualifies, summary=totals))


if __name__ == '__main__': main()
