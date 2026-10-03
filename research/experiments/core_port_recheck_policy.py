"""Training-derived fast policy: recenter only original strong unsupported cues."""
import copy
from core_port_recenter_policy import POLICY as RECENTER_POLICY,proposals as all_proposals,windows,select_zoom,confirmed
POLICY=copy.deepcopy(RECENTER_POLICY)
POLICY.update(proposal_minimum_score=.5,proposal_score_comparison='strictly_greater',
    rationale='All three training gains were original scores>.5; weak-only controls produced no gain')

def proposals(raw):
    # Strong rows are a prefix of the prior descending-score top6. No new seed enters.
    return [p for p in all_proposals(raw) if p['confidence']>.5]
