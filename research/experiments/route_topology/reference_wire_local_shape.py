"""Endpoint-local mapped PCA diagnostics; NEVER wire identity or continuity."""
import numpy as np

POLICY=dict(min_points=16,axis_covariance_ratio_min=2.,axis_difference_limit_degrees=45.,
            elongation_factor_limit=4.,fitted_on_reference_only=True)


def describe(mask,crop_origin=(0.,0.),H=None):
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype!=bool:
        raise ValueError('two-dimensional boolean selection required')
    ys,xs=np.nonzero(mask)
    result=dict(points=len(xs),physical_identity_confirmed=False)
    if len(xs)<POLICY['min_points']:
        return dict(result,state='insufficient_local_shape')
    xy=np.column_stack((xs+crop_origin[0],ys+crop_origin[1])).astype(float)
    if H is not None:
        H=np.asarray(H,dtype=float)
        if H.shape!=(3,3) or not np.isfinite(H).all():
            return dict(result,state='invalid_coordinate_mapping')
        q=np.column_stack((xy,np.ones(len(xy))))@H.T
        if np.any(abs(q[:,2])<1e-9):
            return dict(result,state='invalid_coordinate_mapping')
        xy=q[:,:2]/q[:,2:]
    if not np.isfinite(xy).all():
        return dict(result,state='invalid_coordinate_mapping')
    center=xy.mean(axis=0);delta=xy-center
    covariance=delta.T@delta/len(xy)
    values,vectors=np.linalg.eigh(covariance)
    major=float(max(values[1],0.));minor=float(max(values[0],0.))
    ratio=major/max(minor,1e-9)
    if major<=1e-9:
        return dict(result,state='insufficient_local_shape')
    angle=float(np.degrees(np.arctan2(vectors[1,1],vectors[0,1]))%180.)
    return dict(result,state='usable_local_shape',centroid_xy=center.tolist(),
                covariance=covariance.tolist(),axis_degrees=angle,
                covariance_ratio=ratio,elongation=float(np.sqrt(ratio)),
                axis_usable=bool(ratio>=POLICY['axis_covariance_ratio_min']))


def compare(reference,observed):
    result=dict(physical_identity_confirmed=False,electrical_continuity='not_assessed',
                geometry_mismatch_is_not_confirmed_foreign_wire=True)
    if any(r['state']!='usable_local_shape' for r in [reference,observed]):
        return dict(result,axis_state='insufficient_local_shape',elongation_state='insufficient_local_shape')
    factor=max(reference['elongation'],observed['elongation'])/min(reference['elongation'],observed['elongation'])
    result.update(elongation_factor=factor,elongation_state=(
        'local_elongation_reference_consistent' if factor<=POLICY['elongation_factor_limit'] else 'local_elongation_differs'))
    if not reference['axis_usable'] or not observed['axis_usable']:
        return dict(result,axis_state='insufficient_local_axis')
    difference=abs(reference['axis_degrees']-observed['axis_degrees'])%180.
    difference=min(difference,180.-difference)
    return dict(result,axis_difference_degrees=difference,axis_state=(
        'local_axis_reference_consistent' if difference<=POLICY['axis_difference_limit_degrees'] else 'local_axis_differs'))
