"""Reference-only calibration contract; mapped ROIs are proposals, not connections.

Image hashes bind one reference template. Inspection coordinates are obtained
by inverse registration, never by copying reference coordinates or confirmations.
No GT, filename classification, missing-path completion, or fault verdict.
"""
from copy import deepcopy

import numpy as np

from core import reviewed, text, validate_ports


def validate_binding(binding):
    digest = binding.get('image_sha256')
    size = binding.get('image_size')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
            or binding.get('coordinate_frame') != 'source_image_pixels'):
        raise ValueError('invalid source image binding')


def draft_template(reference_path, binding):
    validate_binding(binding)
    return {'schema_version': 1, 'kind': 'reference_once_visible_topology',
            'reference_image_path': str(reference_path),
            'reference_binding': deepcopy(binding),
            'scope_description': 'Pending actual reference port and visible relation review',
            'ports': [], 'expected_connections': None,
            'reference_review': {'confirmed': False, 'reviewer': None, 'evidence_note': None},
            'claim_boundary': 'Visible scoped connections only; electrical continuity not assessed'}


def validate_template(template, binding):
    validate_binding(binding)
    if (type(template.get('schema_version')) is not int or template['schema_version'] != 1
            or template.get('kind') != 'reference_once_visible_topology'):
        raise ValueError('unsupported reference-only template')
    if template.get('reference_binding') != binding:
        raise ValueError('reference fingerprint/frame mismatch')
    text(template.get('scope_description'), 'scope description')
    identities = validate_ports(template.get('ports'), binding['image_size'])
    confirmed = reviewed(template.get('reference_review'))
    edges = template.get('expected_connections')
    keys = set()
    if edges is not None:
        if not isinstance(edges, list):
            raise ValueError('expected connections must be null or list')
        for edge in edges:
            a, b = text(edge.get('from'), 'from'), text(edge.get('to'), 'to')
            key = tuple(sorted((a, b)))
            if a not in identities or b not in identities or a == b or key in keys:
                raise ValueError('invalid or duplicate reference relation')
            keys.add(key)
    if confirmed and (not keys or any(not reviewed(p) for p in template['ports'])):
        raise ValueError('confirmed reference requires reviewed ports and nonempty relations')
    return confirmed


def map_reference_ports(template, reference_binding, inspection_binding,
                        inspection_to_reference, registration):
    confirmed = validate_template(template, reference_binding)
    validate_binding(inspection_binding)
    if not isinstance(registration, dict) or type(registration.get('reliable')) is not bool:
        raise ValueError('registration reliability must be explicit boolean')
    result = {'schema_version': 1, 'topology_decision': 'insufficient_evidence',
              'reference_confirmed': confirmed, 'port_proposals': [], 'rejections': [],
              'new_confirmed_connections': 0, 'inspection_identities_confirmed': False,
              'electrical_continuity': 'not_assessed', 'automatic_fault_verdict': False,
              'reason': 'inspection_local_port_and_path_evidence_not_yet_verified'}
    if not registration['reliable']:
        result['reason'] = 'registration_not_reliable'
        return result
    matrix = np.asarray(inspection_to_reference, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('finite nonsingular inspection-to-reference transform required')
    inverse = np.linalg.inv(matrix)
    width, height = inspection_binding['image_size']
    for port in template['ports']:
        left, top, right, bottom = port['bbox_xyxy']
        corners = np.array([[left, top, 1], [right, top, 1],
                            [right, bottom, 1], [left, bottom, 1]], dtype=float)
        q = corners @ inverse.T
        # A projective horizon through the ROI invalidates it, even when all
        # four corner divisions happen to produce finite numbers.
        if np.any(np.abs(q[:, 2]) < 1e-9) or not (np.all(q[:, 2] > 0) or np.all(q[:, 2] < 0)):
            result['rejections'].append({'id': port['id'], 'reason': 'projective_horizon_in_roi'})
            continue
        polygon = q[:, :2] / q[:, 2:]
        if (not np.isfinite(polygon).all() or np.any(polygon < 0)
                or np.any(polygon[:, 0] > width) or np.any(polygon[:, 1] > height)):
            result['rejections'].append({'id': port['id'], 'reason': 'mapped_roi_outside_inspection'})
            continue
        result['port_proposals'].append({'id': port['id'], 'roi_kind': 'wire_entry_port',
            'polygon_xy': polygon.tolist(), 'bbox_xyxy': [float(polygon[:, 0].min()),
                float(polygon[:, 1].min()), float(polygon[:, 0].max()), float(polygon[:, 1].max())],
            'confirmed': False, 'reviewer': None, 'evidence_note': None,
            'identity_evidence': 'global_registration_proposal_only',
            'coordinate_frame': 'source_image_pixels',
            'image_binding': deepcopy(inspection_binding)})
    if not template['ports'] or not confirmed:
        result['reason'] = 'reference_calibration_not_completed'
    return result
