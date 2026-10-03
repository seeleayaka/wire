"""Scoped finer physical tiles; same models, input resolution and merge gates."""
import copy
from inspection_agent.optional_port_crop_review import CONFIG,predict


def infer(model,image):
    before = copy.deepcopy(CONFIG)
    try:
        CONFIG['tile_size'] = 960
        CONFIG['tile_stride'] = 720
        return predict(model,image)
    finally:
        CONFIG.clear(); CONFIG.update(before)
