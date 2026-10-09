"""One common physical scale, same three model identities and all gates."""
import copy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path('E:/PythonProject10')))
from inspection_agent.optional_port_crop_review import CONFIG,predict


def infer(model,image):
    before=copy.deepcopy(CONFIG)
    if before['predict_imgsz']!=960:raise ValueError('fixed model input960 required')
    try:
        CONFIG['tile_size']=640;CONFIG['tile_stride']=480
        return predict(model,image)
    finally:
        CONFIG.clear();CONFIG.update(before)
