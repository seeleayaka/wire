"""Plain24 checkpoint with original independent dense-source gates."""
from pathlib import Path
import torch
import evaluate_dense_dino_source_support as evaluator
from dense_port_probe import frozen_features as semantic_features,preprocess
from dense_pixel_port_probe import DensePixelPortHead,decode,GRID
from inspection_agent.optional_port_crop_review import sha


def main():
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('dense_pixel_port_probe.py'),
         Path(__file__).with_name('train_dense_pixel_plain.py'),Path(evaluator.__file__))}
    old=(evaluator.TRAIN,evaluator.OUT,evaluator.DensePortHead,evaluator.decode,evaluator.frozen_features,evaluator.save,evaluator.EPOCHS)
    def features(encoder,images):return dict(semantic=semantic_features(encoder,images),rgb=torch.stack([preprocess(im) for im in images]).to(torch.float16))
    def save(path,value):
        if path.name=='protocol.json':
            value=dict(value);value['pins']=dict(value['pins'],**pins)
            value.update(pixel_localization_branch=True,fine_grid=GRID,fixed_total_training_epochs=24,augmentation=False,original_scoring_and_source_cutoffs_unchanged=True)
        assert {p:sha(Path(p)) for p in pins}==pins
        return old[5](path,value)
    try:
        evaluator.TRAIN=evaluator.ROOT/'artifacts/dense_pixel_plain_20261003';evaluator.OUT=evaluator.TRAIN/'source_acceptance'
        evaluator.DensePortHead=DensePixelPortHead;evaluator.decode=decode;evaluator.frozen_features=features;evaluator.save=save;evaluator.EPOCHS=24
        evaluator.main()
    finally:evaluator.TRAIN,evaluator.OUT,evaluator.DensePortHead,evaluator.decode,evaluator.frozen_features,evaluator.save,evaluator.EPOCHS=old


if __name__=='__main__':main()
