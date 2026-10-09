"""Source-labelled visible connector phenotype; NEVER an electrical verdict."""
import numpy as np

POLICY = {'feature': 'registered_socket_LAB_gradient_320', 'l2': .02,
    'learning_rate': .05, 'iterations': 1000, 'std_floor': .01,
    'class_conditional_alpha': .05, 'source_gate_wrong_singletons_max': 0,
    'source_gate_singleton_coverage_min': .75, 'classification': 'visible_phenotype_only'}


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def fit(features, labels):
    x = np.asarray(features, np.float64); y = np.asarray(labels, np.float64)
    if x.ndim != 2 or x.shape[1] != 320 or y.shape != (len(x),):
        raise ValueError('aligned 320-dimensional source features required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or set(y) != {0., 1.}:
        raise ValueError('finite features and both visual classes required')
    if min((y == k).sum() for k in [0, 1]) < 20:
        raise ValueError('at least20 fit samples per class required')
    center = x.mean(axis=0); scale = np.maximum(x.std(axis=0), POLICY['std_floor'])
    z = (x - center) / scale
    weights = np.array([len(y) / (2 * (y == k).sum()) for k in y])
    w = np.zeros(320); bias = 0.
    for _ in range(POLICY['iterations']):
        error = (sigmoid(z @ w + bias) - y) * weights
        w -= POLICY['learning_rate'] * (z.T @ error / len(y) + POLICY['l2'] * w)
        bias -= POLICY['learning_rate'] * error.mean()
    return {'center': center, 'scale': scale, 'weights': w, 'bias': float(bias)}


def probability(model, features):
    x = np.asarray(features, np.float64)
    if x.shape != (320,) or not np.isfinite(x).all(): raise ValueError('invalid feature')
    return float(sigmoid(((x-model['center']) / model['scale']) @ model['weights'] + model['bias']))


def predict(model, calibration, features):
    p = probability(model, features); ranks = {}
    for k, candidate in [(0, p), (1, 1-p)]:
        scores = np.asarray(calibration[k], np.float64)
        if scores.ndim != 1 or len(scores) < 20 or not np.isfinite(scores).all():
            raise ValueError('at least20 disjoint calibration samples per class required')
        ranks[k] = float((1 + (scores >= candidate).sum()) / (1 + len(scores)))
    admitted = [k for k in [0, 1] if ranks[k] > POLICY['class_conditional_alpha']]
    label = admitted[0] if len(admitted) == 1 else None
    return {'visual_label_candidate': label, 'phenotype': 'mating_body_visible' if label == 1
        else 'socket_contacts_exposed' if label == 0 else 'uncertain',
        'class_tail_ranks': {str(k): v for k, v in ranks.items()},
        'mating_body_model_probability': p, 'decision': 'insufficient_evidence',
        'new_confirmed_connections': 0, 'confirmed_disconnections': 0,
        'electrical_continuity': 'not_assessed', 'port_identity_confirmed': False}


def source_gate(rows):
    singletons = [r for r in rows if r['prediction']['visual_label_candidate'] is not None]
    wrong = sum(r['prediction']['visual_label_candidate'] != r['visual_label'] for r in singletons)
    coverage = len(singletons) / len(rows) if rows else 0.
    return {'passed': wrong == 0 and coverage >= .75, 'wrong_singletons': wrong,
        'singleton_count': len(singletons), 'calibration_count': len(rows),
        'singleton_coverage': coverage, 'not_field_accuracy': True}
