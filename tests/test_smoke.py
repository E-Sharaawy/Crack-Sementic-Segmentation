from pathlib import Path

import torch

from generate_synthetic_data import generate_dataset
from src.data import CrackDataset
from src.model import TinyUNet
from train import segmentation_metrics


def test_model_preserves_spatial_shape() -> None:
    model = TinyUNet(base_channels=4)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 1, 64, 64)


def test_synthetic_dataset_and_metrics(tmp_path: Path) -> None:
    generate_dataset(tmp_path, samples=2, size=32, seed=7)
    images, masks = CrackDataset(tmp_path, image_size=32)[0]
    assert images.shape == (3, 32, 32)
    assert masks.shape == (1, 32, 32)
    dice, iou = segmentation_metrics(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))
    assert dice == 1.0
    assert iou == 1.0

