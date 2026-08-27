"""Dataset utilities for paired crack images and binary masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CrackDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load pairs from ``images/`` and ``masks/`` using matching filenames."""

    def __init__(self, root: str | Path, image_size: int = 128) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.image_size = image_size
        self.image_paths = sorted(
            path for path in self.image_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        missing = [path.name for path in self.image_paths if not (self.mask_dir / path.name).exists()]
        if missing:
            raise ValueError(f"Missing masks for: {', '.join(missing[:5])}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        size = (self.image_size, self.image_size)
        image = Image.open(image_path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
        mask = Image.open(self.mask_dir / image_path.name).convert("L").resize(size, Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor

