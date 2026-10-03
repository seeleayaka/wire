"""New integration regression output; preserve all historical suite reports."""
import os
import run_resolution_project_regression as evaluator


def main():
    original=evaluator.OUT
    try:
        evaluator.OUT=evaluator.ROOT/'artifacts/paired_geometry_project_regression_20261003'
        os.environ['YOLO_CONFIG_DIR']=str(evaluator.OUT/'config')
        evaluator.main()
    finally:evaluator.OUT=original

if __name__=='__main__':main()
