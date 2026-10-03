"""Fresh suite after default-off median geometry integration."""
import os
import run_resolution_project_regression as evaluator


def main():
    previous = evaluator.OUT
    try:
        evaluator.OUT = evaluator.ROOT / 'artifacts/paired_median_project_regression_20261003'
        os.environ['YOLO_CONFIG_DIR'] = str(evaluator.OUT / 'config')
        evaluator.main()
    finally:
        evaluator.OUT = previous


if __name__ == '__main__': main()
