"""Local area density feature, never a connectivity mask or metal identity."""
import cv2
import numpy as np

def score(rgb, support):
    from run_positive_contact_gate import bright_contacts
    support=np.asarray(support)
    if support.dtype!=bool or support.shape!=rgb.shape[:2] or not support.any():
        raise ValueError('nonempty same-frame boolean FIT support required')
    density=cv2.boxFilter(bright_contacts(rgb).astype(np.float32),cv2.CV_32F,
        (5,5),normalize=True,borderType=cv2.BORDER_CONSTANT)
    return float(density[support].mean())
