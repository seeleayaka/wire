"""Extra searched cues require all three distinct current detector checkpoints."""
from paired_pose_search import select as pose_select


def select(current, candidates, probabilities, head_sha):
    pose_select(current, candidates, probabilities, head_sha)  # Validate unfiltered inputs too.
    accepted = [(row,p) for row,p in zip(candidates,probabilities)
        if len(set(row['semantic_model_vote_sha256'])) >= 3]
    return pose_select(current,[row for row,p in accepted],[p for row,p in accepted],head_sha)
