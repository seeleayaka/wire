"""Color/edge disagreement abstention; three views of ONE evidence source."""
import numpy as np

POLICY={'views':['combined','color','edge'],'rule':'all_three_singletons_same_or_abstain',
    'counts_as_independent_observers':1,'no_probability_threshold_change':True,
    'source_gate_wrong_singletons_max':0,'source_gate_singleton_coverage_min':.75,
    'calibration_is_reused_development_not_independent_final_test':True}


def feature_view(features,view):
    x=np.asarray(features,np.float64)
    if x.shape!=(320,) or not np.isfinite(x).all():raise ValueError('finite320 features required')
    if view=='combined':return x.copy()
    selected=np.zeros(320,np.float64)
    mask=np.arange(320)%10<6
    if view=='edge':mask=~mask
    elif view!='color':raise ValueError('unknown feature view')
    selected[mask]=x[mask];return selected


def agreement(predictions):
    if set(predictions)!=set(POLICY['views']):raise ValueError('all frozen feature views required')
    labels=[predictions[k]['visual_label_candidate'] for k in POLICY['views']]
    if any(v is not None and v not in [0,1] for v in labels):raise ValueError('invalid visual label')
    label=labels[0] if labels[0] is not None and all(v==labels[0] for v in labels) else None
    return {'visual_label_candidate':label,'phenotype':'mating_body_visible' if label==1
        else 'socket_contacts_exposed' if label==0 else 'uncertain',
        'feature_view_predictions':predictions,'independent_observer_count':1,
        'decision':'insufficient_evidence','new_confirmed_connections':0,'confirmed_disconnections':0,
        'electrical_continuity':'not_assessed','port_identity_confirmed':False}
