"""Local contact review only: proximity never becomes a connection edge."""
from collections import Counter
import math
from pathlib import Path
from inspection_agent.terminal_mapping import validate_mapping


def review_port_contacts(mapping, image_path, report):
    binding = validate_mapping(mapping, Path(image_path))
    if not isinstance(report, dict) or type(report.get('geometry_contract_version')) is not int or report.get('geometry_contract_version') != 1:
        raise ValueError('geometry contract required')
    supplied = report.get('image_binding', {})
    if not isinstance(supplied, dict):
        raise ValueError('image binding must be an object')
    if any(supplied.get(k) != binding[k] for k in ('image_sha256', 'image_size', 'coordinate_frame')):
        raise ValueError('endpoint image binding mismatch')
    records = report.get('records')
    if not isinstance(records, list):
        raise ValueError('records must be a list')
    if any(not isinstance(record, dict) for record in records):
        raise ValueError('record must be an object')
    ports = sorted(mapping['ports'], key=lambda p: p['id'])
    rows, abstained, seen = [], [], set()
    for record in sorted(records, key=lambda r: str(r.get('record_id'))):
        rid = record.get('record_id')
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError('unique record IDs required')
        seen.add(rid)
        score = record.get('source_score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid source score')
        ends = record.get('visible_ends_xy', [])
        if not isinstance(ends, list):
            raise ValueError('invalid endpoints')
        for point in ends:
            if not isinstance(point, list) or len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
                raise ValueError('invalid endpoint coordinates')
            if not 0 <= point[0] < binding['image_size'][0] or not 0 <= point[1] < binding['image_size'][1]:
                raise ValueError('endpoint outside image')
        if (record.get('geometry_pair_eligible') is not True or len(ends) != 2
                or record.get('candidate_tip_count') != 2 or record.get('branch_cluster_count') != 0
                or record.get('skeleton_component_count') != 1
                or record.get('crop_evidence', {}).get('boundary_truncated') is True):
            abstained.append({'record_id': rid, 'reason': 'geometry_or_crop_evidence_insufficient'})
            continue
        for index, point in enumerate(ends):
            contacts, distances = [], []
            for port in ports:
                x1, y1, x2, y2 = port['bbox_xyxy']
                # Half-open boxes are consistent with image pixel ROI slicing.
                if x1 <= point[0] < x2 and y1 <= point[1] < y2:
                    contacts.append(port['id'])
                dx = max(x1-point[0], 0, point[0]-x2)
                dy = max(y1-point[1], 0, point[1]-y2)
                distances.append({'port_id': port['id'], 'distance_to_roi_closure_px': math.hypot(dx, dy)})
            rows.append({'record_id': rid, 'endpoint_index': index, 'point_xy': point[:],
                         'source_score': score, 'contact_port_ids': contacts,
                         'nearest_diagnostics_not_assignments': sorted(distances, key=lambda d: (d['distance_to_roi_closure_px'], d['port_id']))[:3]})
    usage = Counter(pid for row in rows for pid in row['contact_port_ids'])
    confirmed = {p['id']: p['confirmed'] for p in ports}
    for row in rows:
        ids = row['contact_port_ids']
        if not ports:
            state = 'port_map_missing'
        elif not ids:
            state = 'no_roi_contact_not_missing_wire'
        elif len(ids) > 1:
            state = 'ambiguous_overlapping_rois'
        elif usage[ids[0]] > 1:
            state = 'ambiguous_shared_port_evidence'
        elif not confirmed[ids[0]]:
            state = 'draft_roi_contact_unconfirmed'
        else:
            state = 'local_contact_manual_review'
        row['state'] = state
        row['confirmed_assignment'] = False
    return {'schema_version': 1, 'map_id': mapping['map_id'], 'image_binding': binding,
            'rows': rows, 'abstained_records': abstained,
            'summary': dict(sorted(Counter(row['state'] for row in rows).items())),
            'decision': 'insufficient_evidence', 'automatic_connections_emitted': False,
            'connection_edges': [], 'manual_confirmation_required': True,
            'claim_boundary': 'ROI contact is not conductor continuity, seating or cable identity.'}
