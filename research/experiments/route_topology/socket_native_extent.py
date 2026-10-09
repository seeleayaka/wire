"""Socket contradiction needs one native component reaching beyond the socket.

Port-contained pin speckles are not a visible outgoing wire. All raw pixels,
components, and the old any-touch flag remain retained; no dilation/bridging.
The spatial unit is the reviewed port's shorter side, not image-specific pixels.
This is a post-review development hypothesis, NOT validated semantic identity.
"""
import cv2
import numpy as np
from visible_bundle_relation import observe_bundle, compare_bundle

POLICY = {'kind': 'socket_native_outgoing_component_v1', 'score_min': .75,
          'outside_extent_port_short_sides': 1.0, 'connectivity': 8,
          'same_native_component_must_touch_and_exit': True,
          'pixel_modification': False, 'model_observers': 1,
          'semantic_wire_identity_certified': False}


def socket_extent(raw, local_transform, socket_box, translation=(0, 0)):
    raw = np.asarray(raw)
    transform = np.asarray(local_transform, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('finite native 2D mask required')
    if transform.shape != (3, 3) or not np.isfinite(transform).all() or np.linalg.matrix_rank(transform) != 3:
        raise ValueError('qualified nonsingular local mapping required')
    left, top, right, bottom = socket_box
    unit = min(right-left, bottom-top)
    if unit <= 0 or not np.isfinite(socket_box).all():
        raise ValueError('valid reviewed socket rectangle required')
    if len(translation) != 2 or any(type(v) is not int or v < 0 for v in translation):
        raise ValueError('exact nonnegative crop translation required')
    count, labels = cv2.connectedComponents((raw > 0).astype(np.uint8), connectivity=8)
    rows = []
    for index in range(1, count):
        ys, xs = np.nonzero(labels == index)
        points = np.column_stack((xs+translation[0], ys+translation[1], np.ones(len(xs))))
        q = points @ transform.T
        if np.any(np.abs(q[:, 2]) < 1e-9) or not (np.all(q[:, 2] > 0) or np.all(q[:, 2] < 0)):
            raise ValueError('projective horizon crosses a native component')
        mapped = q[:, :2] / q[:, 2:]
        x, y = mapped.T
        hits = int(((x >= left) & (x <= right) & (y >= top) & (y <= bottom)).sum())
        distances = np.maximum.reduce([left-x, x-right, top-y, y-bottom, np.zeros(len(x))])
        extent = float(np.max(distances)/unit)
        rows.append({'component_index': index, 'pixel_count': len(xs), 'socket_pixel_support': hits,
                     'outside_extent_port_short_sides': extent,
                     'touches_socket_and_reaches_outgoing_context': bool(hits > 0 and extent >= 1.0)})
    return rows


def observe_with_extent(scope, binding, poses, masks, socket_state, translation=(0, 0), sam_inventory_verified=False):
    observation = observe_bundle(scope, binding, poses, masks, socket_state, translation, sam_inventory_verified)
    socket = next(a for a in scope['anchors'] if a['kind'] == 'wire_entry_socket')
    pose = next((p for p in poses if p['id'] == socket['id']), None)
    audit, qualified = [], []
    if observation['local_anchor_proposals_supported'].get(socket['id'], False):
        for item in masks:
            rows = socket_extent(item['raw'], pose['inspection_to_reference_local'], socket['bbox_xyxy'], translation)
            audit.append({'record_id': item['record_id'], 'score': item['score'], 'components': rows})
            for row in rows:
                if item['score'] >= .75 and row['touches_socket_and_reaches_outgoing_context']:
                    qualified.append({'record_id': item['record_id'], 'component_index': row['component_index']})
    observation.update(socket_outgoing_policy=POLICY, socket_outgoing_component_audit=audit,
                       native_outgoing_component_at_socket=bool(qualified),
                       native_outgoing_components=qualified)
    return observation


def compare_with_extent(scope, reference, inspection):
    result = compare_bundle(scope, reference, inspection)
    if inspection.get('socket_outgoing_policy') != POLICY:
        raise ValueError('audited native outgoing-component policy required')
    result['socket_contradiction_policy'] = POLICY['kind']
    if result.get('reason') == 'exposed_socket_and_high_score_mask_conflict' and not inspection['native_outgoing_component_at_socket']:
        result.update(decision='visible_socket_attachment_change_supported',
                      reason='positive_exposed_socket_without_native_outgoing_wire_context',
                      fan_end_relationship_assessed=False,
                      old_any_touch_conflict_retained=True,
                      semantic_wire_identity_certified=False)
    return result
