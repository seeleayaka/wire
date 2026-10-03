"""Train a small YOLO-seg teacher used only for cable pre-label proposals."""

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
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=project_root / "output")
    parser.add_argument("--name", default="mendeley_cable_prelabel_teacher")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if not args.data.is_file() or not args.weights.is_file():
        raise FileNotFoundError("data YAML or segmentation weights are missing")
    if torch.cuda.is_available():
        raise RuntimeError("This pre-label trial is intentionally CPU-only.")

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
        freeze=0,
        optimizer="SGD",
        lr0=0.001,
        lrf=0.01,
        cos_lr=False,
        amp=False,
        cache=False,
        deterministic=True,
        seed=20260825,
        patience=0,
        mosaic=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.15,
        fliplr=0.5,
        flipud=0.0,
        val=False,
        plots=True,
        save=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
