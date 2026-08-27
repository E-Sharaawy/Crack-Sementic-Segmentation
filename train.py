"""Train TinyUNet and report validation Dice and IoU."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data import CrackDataset
from src.model import TinyUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prediction = torch.sigmoid(logits) >= 0.5
    target = target >= 0.5
    intersection = (prediction & target).sum(dim=(1, 2, 3)).float()
    predicted_pixels = prediction.sum(dim=(1, 2, 3)).float()
    target_pixels = target.sum(dim=(1, 2, 3)).float()
    union = (prediction | target).sum(dim=(1, 2, 3)).float()
    dice = ((2 * intersection + 1) / (predicted_pixels + target_pixels + 1)).mean()
    iou = ((intersection + 1) / (union + 1)).mean()
    return dice.item(), iou.item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    dice_scores, iou_scores = [], []
    for images, masks in loader:
        logits = model(images.to(device))
        dice, iou = segmentation_metrics(logits, masks.to(device))
        dice_scores.append(dice)
        iou_scores.append(iou)
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = CrackDataset(args.data, image_size=args.image_size)
    validation_size = max(1, int(len(dataset) * args.validation_fraction))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("The dataset needs at least two samples.")
    train_set, validation_set = random_split(
        dataset,
        (train_size, validation_size),
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=0)

    model = TinyUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.positive_weight], device=device))
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, masks in progress:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        dice, iou = evaluate(model, validation_loader, device)
        print(f"epoch={epoch:02d} loss={np.mean(losses):.4f} val_dice={dice:.4f} val_iou={iou:.4f}")
        if dice > best_dice:
            best_dice = dice
            torch.save({"model_state": model.state_dict(), "base_channels": args.base_channels}, checkpoint_path)

    print(f"Best checkpoint saved to {checkpoint_path} (Dice={best_dice:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_unet.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())

