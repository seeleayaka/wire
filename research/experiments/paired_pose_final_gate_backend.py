"""Scoped pose proposal adapter through unchanged original reference/ROI gates."""
import copy
from pathlib import Path
from unittest.mock import patch
import paired_median_geometry_backend as backend
from paired_pose_search import proposals,select
from prepare_paired_port_semantics import REPO,sha
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
POLICY = 'median_preserved_uniform_pose_original_final_gates_prototype_20261003'


def append_pose_review(report,original,*,project):
    frozen = median_runtime_fingerprint(project)
    evidence = original.get('median_geometry_evidence') or original.get('paired_geometry_evidence')
    if not evidence:
        result = copy.deepcopy(original)
        result['pose_geometry_policy'] = dict(policy_id=POLICY,enabled=True,added_hints=0,
            upstream_without_native_evidence=True,automatic_fault_verdict=False,model_deployed=False)
        return result
    current = copy.deepcopy(evidence['native'])
    if len(original.get('supplementary_hints',[])) >= 5 or len(current['all_predictions'])-len(current['primary']) >= 5:
        result = copy.deepcopy(original)
        result['pose_geometry_policy'] = dict(policy_id=POLICY,enabled=True,added_hints=0,
            native_candidates=0,shared_budget_full=True,automatic_fault_verdict=False,model_deployed=False)
        return result
    pins = {str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_pose_search.py'),Path(backend.__file__))}
    selected_call = {}
    def scoped_proposals(teacher,models): return proposals(teacher,models,current)
    def scoped_select(parent,native,probabilities,head_sha):
        if parent['all_predictions'] != current['all_predictions'][:len(parent['all_predictions'])]:
            raise ValueError('Actual median must preserve original paired native prefix')
        fused = select(current,native,probabilities,head_sha)
        selected_call['native'] = fused
        return fused
    with patch.object(backend,'proposals',scoped_proposals),patch.object(backend,'select',scoped_select):
        result = backend.append_median_review(report,original,project=project)
    policy = result.pop('median_geometry_policy'); policy.update(policy_id=POLICY,source_runtime_fingerprint=frozen,adapter_pins=pins)
    result['pose_geometry_policy'] = policy
    new_evidence = result.pop('median_geometry_evidence',None)
    if new_evidence is not None and selected_call.get('native') == new_evidence.get('native'):
        new_evidence['native_current'] = current
        result['pose_geometry_evidence'] = new_evidence
    if 'median_geometry_policy' in original: result['median_geometry_policy'] = copy.deepcopy(original['median_geometry_policy'])
    if 'median_geometry_evidence' in original: result['median_geometry_evidence'] = copy.deepcopy(original['median_geometry_evidence'])
    for key,value in original.items():
        if key == 'supplementary_hints': assert result[key][:len(value)] == value
        else: assert result[key] == value
    assert {p:sha(Path(p)) for p in pins} == pins and median_runtime_fingerprint(project) == frozen
    return result
