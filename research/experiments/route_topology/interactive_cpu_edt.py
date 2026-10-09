"""Process-local CPU EDT adapter, never replaces model or segmentation math.

The upstream EDT documents equivalence to cv2 DIST_L2/MASK_PRECISE.
No source/environment writes; unsupported devices and ranks fail closed.
"""
import sys
import types

CALL_COUNT = 0


def edt_cpu(data):
    import cv2
    import numpy as np
    import torch
    global CALL_COUNT
    if data.device.type != 'cpu' or data.ndim != 3:
        raise ValueError('CPU EDT adapter requires B,H,W CPU tensors')
    CALL_COUNT += 1
    outputs = []
    for sample in data.detach().numpy():
        active = (sample != 0).astype(np.uint8)
        # Upstream finite infinity is 1e18 before sqrt.
        result = (np.full(active.shape, 1e9, np.float32) if active.all()
                  else cv2.distanceTransform(active, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        outputs.append(torch.from_numpy(result))
    return torch.stack(outputs)


def install():
    name = 'sam3.model.edt'
    if name in sys.modules:
        raise RuntimeError('refuse replacing an already imported EDT module')
    module = types.ModuleType(name)
    module.__file__ = __file__
    module.edt_triton = edt_cpu
    module.adapter_backend = 'opencv_precise_cpu_explicit_research_adapter'
    sys.modules[name] = module
