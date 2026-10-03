"""Quality-trained24 checkpoint, original unchanged dense source gates."""
from pathlib import Path
import evaluate_dense_pixel_plain_source as plain
from inspection_agent.optional_port_crop_review import sha


def main():
    e=plain.evaluator
    oldmain=e.main
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('dense_quality_objective.py'),Path(__file__).with_name('train_dense_quality.py'))}
    original_save=e.save
    def check():
        e.TRAIN=e.ROOT/'artifacts/dense_quality_20261003';e.OUT=e.TRAIN/'source_acceptance'
        current_save=e.save
        def save(path,value):
            assert {p:sha(Path(p)) for p in pins}==pins
            if path.name=='protocol.json':
                value=dict(value);value['pins']=dict(value['pins'],**pins);value['joint_localization_quality_training']=True
            return current_save(path,value)
        e.save=save
        try:return oldmain()
        finally:e.save=current_save
    try:e.main=check;plain.main()
    finally:e.main=oldmain;e.save=original_save


if __name__=='__main__':main()
