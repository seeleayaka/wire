"""Run a bounded CPU-only training gate for visible unplugged ports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path(r"E:\wire_harness_training_bundle\weights\yolov8s-seg.pt"))
    parser.add_argument("--project", type=Path, default=project_root / "output" / "mendeley_port_state_cpu_poc_20260825")
    parser.add_argument("--name", default="yolov8s_rectports_cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if not args.data.is_file() or not args.weights.is_file():
        raise FileNotFoundError("data YAML or verified local segmentation base weight is missing")
    if torch.cuda.is_available():
        raise RuntimeError("this gate is intentionally CPU-only; choose a separate GPU run explicitly")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=1,
        workers=0,
        device="cpu",
        pretrained=True,
        freeze=10,
        optimizer="SGD",
        lr0=0.002,
        lrf=0.01,
        cos_lr=False,
        amp=False,
        cache=False,
        deterministic=True,
        seed=20260825,
        patience=2,
        mosaic=0.0,
        degrees=0.0,
        translate=0.02,
        scale=0.05,
        fliplr=0.0,
        flipud=0.0,
        plots=True,
        save=True,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
