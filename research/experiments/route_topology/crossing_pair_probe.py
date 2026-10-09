"""Independent minimal curvature-inspired pairing, not a mBEST reproduction.

Only four observed incident tangents are admissible. Missing segments remain
missing; geometrical pairing is a hypothesis, never electrical connection proof.
"""
import itertools
import numpy as np

def pair_tangents(tangents):
    a=np.asarray(tangents,dtype=float)
    base=dict(status='insufficient_evidence',model_observer_count=0,
              new_confirmed_connections=0,electrical_continuity='not_assessed',
              inferred_pairing_is_observed_connection=False)
    if a.shape!=(4,2) or not np.isfinite(a).all(): return dict(base,reason='four_observed_arms_required')
    norms=np.linalg.norm(a,axis=1)
    if (norms<1e-8).any(): return dict(base,reason='invalid_tangent')
    a=a/norms[:,None]
    options=[]
    for first in range(1,4):
        remaining=[i for i in range(1,4) if i!=first]
        pairs=[(0,first),tuple(remaining)]
        cost=float(sum(1+float(a[i]@a[j]) for i,j in pairs))
        options.append((cost,pairs))
    options.sort(key=lambda o:o[0]); margin=options[1][0]-options[0][0]
    if options[0][0]>.25 or margin<.5:
        return dict(base,reason='ambiguous_or_high_curvature',cost=options[0][0],margin=margin)
    return dict(base,status='geometric_pairing_hypothesis',pairs=options[0][1],
                cost=options[0][0],margin=margin)
