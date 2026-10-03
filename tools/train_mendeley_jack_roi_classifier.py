"""Train a small CPU-friendly local classifier for fixed-Dell empty-jack ROIs."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class CropRecord:
    crop: str
    label: int
    roi_index: int


def load_records(dataset_dir: Path, split: str) -> list[CropRecord]:
    records = []
    for line in (dataset_dir / "index.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["split"] == split:
            records.append(CropRecord(item["crop"], int(item["label"]), int(item["roi_index"])))
    if not records:
        raise ValueError(f"no {split} records in {dataset_dir}")
    return records


class RoiCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset_dir: Path, records: list[CropRecord]):
        self.dataset_dir = dataset_dir
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = cv2.imdecode(np.fromfile(str(self.dataset_dir / record.crop), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read crop {record.crop}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        return tensor, torch.tensor(record.label, dtype=torch.long), torch.tensor(record.roi_index, dtype=torch.long)


class TinyRoiClassifier(nn.Module):
    """A deliberately small model; ROI identity is valid for this one fixed chassis."""

    def __init__(self, roi_count: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.roi_embedding = nn.Embedding(roi_count, 12)
        self.head = nn.Sequential(nn.Linear(108, 48), nn.ReLU(inplace=True), nn.Dropout(0.15), nn.Linear(48, 2))

    def forward(self, images: torch.Tensor, roi_indices: torch.Tensor) -> torch.Tensor:
        features = self.features(images).flatten(1)
        return self.head(torch.cat([features, self.roi_embedding(roi_indices)], dim=1))


def class_weights(records: list[CropRecord]) -> torch.Tensor:
    counts = np.bincount([record.label for record in records], minlength=2)
    if not counts.all():
        raise ValueError(f"both classes must be present, got {counts.tolist()}")
    return torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)


def probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets, scores = [], []
    with torch.no_grad():
        for images, labels, roi_indices in loader:
            logits = model(images.to(device), roi_indices.to(device))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return np.asarray(targets, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def choose_threshold(targets: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    """Choose a validation-only threshold; tie-break towards fewer false alarms."""
    best: tuple[float, dict[str, float]] | None = None
    for threshold in np.linspace(0.1, 0.9, 17):
        predicted = scores >= threshold
        tp = int(np.logical_and(predicted, targets == 1).sum())
        fp = int(np.logical_and(predicted, targets == 0).sum())
        fn = int(np.logical_and(~predicted, targets == 1).sum())
        tn = int(np.logical_and(~predicted, targets == 0).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}
        if best is None or (f1, -fp, threshold) > (best[1]["f1"], -best[1]["fp"], best[0]):
            best = (float(threshold), metrics)
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    recipe = json.loads((args.dataset_dir / "recipe.json").read_text(encoding="utf-8"))
    roi_count = len(recipe["candidate_rois"])
    if roi_count < 1:
        raise ValueError("recipe has no candidate ROIs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    train_records, val_records = load_records(args.dataset_dir, "train"), load_records(args.dataset_dir, "val")
    train_loader = DataLoader(RoiCropDataset(args.dataset_dir, train_records), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(RoiCropDataset(args.dataset_dir, val_records), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TinyRoiClassifier(roi_count).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_records).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output.mkdir(parents=True)
    history: list[dict[str, object]] = []
    best_f1, best_payload = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels, roi_indices in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device), roi_indices.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        targets, scores = probabilities(model, val_loader, device)
        threshold, metrics = choose_threshold(targets, scores)
        entry = {"epoch": epoch, "train_loss": loss_sum / len(train_records), "validation_threshold": threshold, "validation": metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_payload = {
                "model_state": model.state_dict(),
                "roi_count": roi_count,
                "roi_ids": [item["candidate_id"] for item in recipe["candidate_rois"]],
                "recommended_threshold": threshold,
                "validation": metrics,
                "recipe": str(args.dataset_dir / "recipe.json"),
            }
            torch.save(best_payload, args.output / "best.pt")
    assert best_payload is not None
    (args.output / "training_report.json").write_text(
        json.dumps({"scope": recipe["purpose"] + "; not seating or continuity verification", "history": history, "best": best_payload["validation"], "recommended_threshold": best_payload["recommended_threshold"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "best": best_payload["validation"], "threshold": best_payload["recommended_threshold"]}))


if __name__ == "__main__":
    main()
