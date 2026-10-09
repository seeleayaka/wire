"""Automatic correspondence proposals for visible segment endpoints.

This is deliberately NOT a port-identity or electrical topology engine. A
unique pair within a coordinate tolerance only nominates the same endpoint
neighborhoods. It never fills confirmed ports or suppresses original cues.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

import numpy as np
from scipy.spatial import cKDTree


def project(points, transform):
    p = np.asarray(points, dtype=float)
    h = np.asarray(transform, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or h.shape != (3,3) or not np.isfinite(p).all() or not np.isfinite(h).all():
        raise ValueError('finite Nx2 points and 3x3 matrix required')
    q = np.column_stack((p, np.ones(len(p)))) @ h.T
    if np.any(np.abs(q[:,2]) < 1e-9):
        raise ValueError('point projects to infinity')
    return q[:,:2] / q[:,2:]


def eligible_records(records):
    result, rejected, duplicates = [], [], []
    by_digest = {}
    ids=[record['record_id'] for record in records]
    if any(not isinstance(identity,str) or not identity for identity in ids) or len(set(ids))!=len(ids):
        raise ValueError('record IDs must be unique nonempty strings within a view')
    for record in sorted(deepcopy(records), key=lambda r:r['record_id']):
        score = record['score']
        if isinstance(score, bool) or not isinstance(score, (int,float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid segmentation score')
        geometry = record['geometry']
        reason = ('invalid_geometry' if geometry.get('geometry_contract_version') != 2 or geometry['state'] != 'simple_visible_path' else
                  'image_boundary_truncation' if record['boundary_truncated'] else
                  'low_segmentation_score' if score < .75 else None)
        if reason:
            rejected.append({'record_id':record['record_id'], 'reason':reason})
            continue
        digest = record['mask_array_sha256']
        path=np.asarray(geometry['path_xy'],float)
        tips=np.asarray(geometry['tips_xy'],float)
        if (path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or tips.shape!=(2,2)
            or not np.isfinite(path).all() or not np.isfinite(tips).all()
            or not (np.array_equal(path[[0,-1]],tips) or np.array_equal(path[[0,-1]],tips[::-1]))):
            raise ValueError('eligible path must have finite coordinates and its actual two tips')
        if digest in by_digest:
            duplicates.append({'record_id':record['record_id'], 'same_mask_as':by_digest[digest]})
        else:
            by_digest[digest] = record['record_id']
            result.append(record)
    return result, rejected, duplicates


def propose_correspondences(reference, inspection, transform, registration, image_size):
    width, height = image_size
    if any(type(v) is not int or v<=0 for v in image_size):
        raise ValueError('invalid frame size')
    if not isinstance(registration, dict) or type(registration.get('reliable')) is not bool:
        raise ValueError('registration reliability must be explicitly declared')
    report = {'schema_version':1, 'decision':'insufficient_evidence', 'proposals':[],
              'reference_rejections':[], 'inspection_rejections':[],
              'automatic_connections':[], 'confirmed_port_identities':[],
              'electrical_continuity':'not_assessed', 'automatic_fault_verdict':False,
              'original_visual_cues_suppressed':False,
              'claim_boundary':'Unlabelled endpoint-neighborhood correspondence proposals, not actual terminal identity or verified wiring.'}
    if not registration['reliable']:
        report['reason'] = 'registration_not_reliable'
        return report
    # Validate even an empty inventory, so a corrupt registered frame cannot
    # quietly pass simply because no instance happened to be present.
    matrix=np.asarray(transform,float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)<3:
        raise ValueError('finite nonsingular homography required')
    refs, ref_rejected, ref_dup = eligible_records(reference)
    ins, ins_rejected, ins_dup = eligible_records(inspection)
    report.update(reference_rejections=ref_rejected, inspection_rejections=ins_rejected,
                  reference_exact_duplicates=ref_dup, inspection_exact_duplicates=ins_dup)
    # Fixed resolution-normalized diagnostic tolerance, never a terminal ROI
    # enlargement or acceptance-threshold substitution for the formal engine.
    radius = .005 * math.hypot(width,height)
    report['endpoint_neighborhood_radius_px'] = radius
    candidates = defaultdict(list)
    inverse_candidates = defaultdict(list)
    distances = {}
    mapped_pair_rejections=[]
    for ri, ref in enumerate(refs):
        a = np.asarray(ref['geometry']['tips_xy'], float)
        for ii, other in enumerate(ins):
            b = project(other['geometry']['tips_xy'], transform)
            options = [np.linalg.norm(a-b,axis=1), np.linalg.norm(a-b[::-1],axis=1)]
            valid_orders = [index for index,d in enumerate(options) if (d <= radius).all()]
            if len(valid_orders) != 1:
                continue
            candidates[ri].append(ii)
            inverse_candidates[ii].append(ri)
            distances[(ri,ii)] = options[valid_orders[0]].tolist()
    for ri, ref in enumerate(refs):
        partners = candidates[ri]
        if len(partners) != 1:
            continue
        ii = partners[0]
        if len(inverse_candidates[ii]) != 1:
            continue
        other = ins[ii]
        ref_path = np.asarray(ref['geometry']['path_xy'],float)
        ins_path = project(other['geometry']['path_xy'],transform)
        # Never extrapolate one hidden/out-of-frame path into a visible match.
        if np.any(ins_path[:,0]<0) or np.any(ins_path[:,0]>=width) or np.any(ins_path[:,1]<0) or np.any(ins_path[:,1]>=height):
            mapped_pair_rejections.append({'reference_record_id':ref['record_id'],
                'inspection_record_id':other['record_id'],'reason':'mapped_path_outside_reference_frame'})
            continue
        forward = cKDTree(ref_path).query(ins_path)[0]
        reverse = cKDTree(ins_path).query(ref_path)[0]
        displacement = max(float(np.quantile(forward,.95)),float(np.quantile(reverse,.95)))
        report['proposals'].append({'reference_record_id':ref['record_id'],
           'inspection_record_id':other['record_id'], 'endpoint_displacements_px':distances[(ri,ii)],
           'path_displacement_p95_px':displacement, 'path_shape_difference_detected':displacement > radius,
           'proposal':'similar_endpoint_neighborhoods_different_visible_route' if displacement > radius else 'similar_visible_segment',
           'confirmed':False, 'port_identity_known':False, 'independent_model_votes':1})
    report.update(reference_eligible=len(refs),inspection_eligible=len(ins),
                  mapped_pair_rejections=mapped_pair_rejections,
                  reference_without_unique_partner=sum(len(candidates[i])!=1 or len(inverse_candidates[candidates[i][0]])!=1 for i in range(len(refs))),
                  inspection_without_unique_partner=sum(len(inverse_candidates[i])!=1 or len(candidates[inverse_candidates[i][0]])!=1 for i in range(len(ins))))
    if report['proposals']:
        report['decision']='visible_segment_correspondence_proposals'
    else:
        report['reason']='no_unique_reliable_visible_segment_pair'
    return report
