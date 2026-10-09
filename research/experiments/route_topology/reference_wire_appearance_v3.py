"""Finite width geometry even for image-filling masks; old variants unchanged."""
import numpy as np
from reference_wire_appearance_v2 import fit_reference as fit_v2,assess as assess_v2,POLICY as PREVIOUS_POLICY

POLICY=dict(PREVIOUS_POLICY,width_zero_border_padding=True)


def fit_reference(rgb,selection):
    result=fit_v2(np.pad(rgb,((1,1),(1,1),(0,0))),np.pad(selection,1))
    result['policy']=POLICY.copy()
    return result


def assess(rgb,mask,reference,coordinate_scale=1.):
    result,pixels=assess_v2(np.pad(rgb,((1,1),(1,1),(0,0))),np.pad(mask,1),reference,coordinate_scale)
    return result,pixels[1:-1,1:-1]
