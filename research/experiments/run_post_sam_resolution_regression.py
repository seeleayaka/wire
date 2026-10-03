"""Preserve the first regression report; add formal final-slot corner tests."""
import os
from pathlib import Path
import run_resolution_project_regression as evaluator
from inspection_agent.optional_port_crop_review import sha


def main():
    before=sha(Path(evaluator.__file__))
    old=evaluator.OUT
    try:
        evaluator.OUT=evaluator.ROOT/'artifacts/resolution_project_regression_post_sam_20261003'
        os.environ['YOLO_CONFIG_DIR']=str(evaluator.OUT/'config')
        evaluator.main()
        assert sha(Path(evaluator.__file__))==before
    finally:evaluator.OUT=old


if __name__=='__main__':main()
