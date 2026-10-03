"""Workspace-only new native head, original real reference/ROI safety gates."""
import copy
from pathlib import Path
from unittest.mock import patch
import paired_pose_final_gate_backend as original_pose
from paired_pose_three_vote import select
from prepare_paired_port_semantics import ROOT,REPO,load,sha
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
HEAD=ROOT/'artifacts/paired_pose_native_training_20261004/full/last_head.pt'
HEAD_SHA='9459ba29715a9f511ddc48a9a99014d2a239cfe65ccc32fe5e2de9122ab72467'
SOURCE=ROOT/'artifacts/paired_pose_native_three_20261004/report.json'
AUDIT=ROOT/'artifacts/paired_pose_native_three_audit_20261004/report.json'
POLICY='median_preserved_native_learned_three_checkpoint_pose_prototype_20261004'


def validate_checkpoint(checkpoint,frozen):
    if checkpoint.get('input_dimensions')!=6144 or checkpoint.get('classes')!=['other','unplugged_plug','unplugged_jack']:
        raise ValueError('Native head contract mismatch')
    if checkpoint.get('encoder_sha256')!=frozen['encoder']:
        raise ValueError('Native encoder contract mismatch')


def scored_features(current,native,old_probabilities,vectors,head):
    if vectors.shape!=(len(native),6144):raise ValueError('Native crop-feature identity count mismatch')
    import torch
    with torch.inference_mode():scores=head(vectors).softmax(dim=1).tolist()
    return select(current,native,scores,HEAD_SHA),scores


def append_native_pose_review(report,original,*,project):
    import torch
    source=load(SOURCE);audit=load(AUDIT)
    if source['status']!='source_pass_requires_reference_ROI_SAM_Qt' or audit['status']!='complete':
        raise ValueError('Full source and independent audit gates required')
    frozen=median_runtime_fingerprint(project)
    if frozen!=source['median_runtime_fingerprint']:raise ValueError('Accepted prefix runtime changed')
    if sha(HEAD)!=HEAD_SHA:raise ValueError('Native head drift')
    checkpoint=torch.load(HEAD,map_location='cpu',weights_only=True)
    validate_checkpoint(checkpoint,frozen)
    head=torch.nn.Linear(6144,3);head.load_state_dict(checkpoint['state_dict'],strict=True);head.eval().requires_grad_(False)
    pins={str(p):sha(p) for p in (HEAD,SOURCE,AUDIT,Path(__file__),Path(original_pose.__file__),
        Path(original_pose.backend.__file__),Path(__file__).with_name('paired_pose_three_vote.py'))}
    captured={};feature_function=original_pose.backend.paired_features
    def capture(observed,expected):
        if 'vectors' in captured:raise ValueError('Multiple native feature batches unexpectedly share selector')
        captured['vectors']=feature_function(observed,expected)
        return captured['vectors']
    def classify(current,native,old_probabilities,old_sha):
        if 'vectors' not in captured:raise ValueError('Missing exact native feature batch')
        fused,scores=scored_features(current,native,old_probabilities,captured['vectors'],head)
        captured.update(native=copy.deepcopy(native),scores=scores,original_scores=copy.deepcopy(old_probabilities),fused=fused)
        return fused
    with patch.object(original_pose.backend,'paired_features',capture),patch.object(original_pose,'select',classify):
        result=original_pose.append_pose_review(report,original,project=project)
    policy=result['pose_geometry_policy'];policy.update(policy_id=POLICY,head_sha256=HEAD_SHA,
        original_head_only_for_preserved_prefix=True,min_distinct_checkpoint_votes=3,native_head_pins=pins,
        model_deployed=False,automatic_fault_verdict=False)
    evidence=result.get('pose_geometry_evidence')
    if evidence:
        assert captured.get('fused')==evidence['native'] and captured['native']==evidence['proposals']
        evidence.update(original_probabilities=captured['original_scores'],probabilities=captured['scores'],
            head_sha256=HEAD_SHA,native_head_pins=pins,accepted_prefix_runtime_fingerprint=frozen)
        for hint in result['supplementary_hints'][len(original['supplementary_hints']):]:
            hint.update(evidence_tier='native_learned_pose_three_checkpoint_manual_review',paired_geometry_head_sha256=HEAD_SHA,
                median_geometry_policy_id=POLICY,automatic_fault_verdict=False)
    for key,value in original.items():
        if key=='supplementary_hints':assert result[key][:len(value)]==value
        else:assert result[key]==value
    assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(project)==frozen
    return result
