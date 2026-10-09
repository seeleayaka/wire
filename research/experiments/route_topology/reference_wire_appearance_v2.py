"""Reference-only color coverage guards against sparse colored reflections."""
from reference_wire_appearance import fit_reference as previous_fit, assess as previous_assess, POLICY as PREVIOUS_POLICY

POLICY=dict(PREVIOUS_POLICY,minimum_colored_coverage_reference_factor=.5)


def fit_reference(rgb,selection):
    profile=previous_fit(rgb,selection)
    profile['minimum_colored_fraction']=.5*min(s['colored_pixels']/int(selection.sum())
                                              for s in profile['reference_samples'])
    profile['policy']=POLICY.copy()
    return profile


def assess(rgb,mask,reference,coordinate_scale=1.):
    metrics,pixels=previous_assess(rgb,mask,reference,coordinate_scale)
    metrics['minimum_colored_fraction']=reference['minimum_colored_fraction']
    metrics['colored_coverage_supported']=metrics['colored_fraction']>=reference['minimum_colored_fraction']
    if metrics['state']=='reference_color_supported' and not metrics['colored_coverage_supported']:
        metrics['state']='insufficient_colored_coverage'
    return metrics,pixels
