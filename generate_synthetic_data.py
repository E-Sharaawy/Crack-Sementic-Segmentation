"""Generate a tiny paired dataset so the repository runs without downloads."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_sample(size: int, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    background = np.clip(rng.gauss(175, 12) + np.random.default_rng(rng.randrange(2**32)).normal(0, 18, (size, size, 3)), 0, 255)
    image = Image.fromarray(background.astype(np.uint8), mode="RGB")
    mask = Image.new("L", (size, size), color=0)
    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    x, y = rng.randrange(size), rng.randrange(size)
    points = [(x, y)]
    for _ in range(rng.randint(4, 9)):
        x = min(size - 1, max(0, x + rng.randint(-size // 4, size // 4)))
        y = min(size - 1, max(0, y + rng.randint(-size // 4, size // 4)))
        points.append((x, y))

    width = rng.randint(1, 3)
    image_draw.line(points, fill=(35, 35, 35), width=width)
    mask_draw.line(points, fill=255, width=max(2, width))
    return image.filter(ImageFilter.GaussianBlur(radius=0.35)), mask


def generate_dataset(output: str | Path, samples: int, size: int, seed: int) -> None:
    output = Path(output)
    image_dir, mask_dir = output / "images", output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for index in range(samples):
        image, mask = generate_sample(size, rng)
        filename = f"sample_{index:04d}.png"
        image.save(image_dir / filename)
        mask.save(mask_dir / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output, args.samples, args.size, args.seed)
    print(f"Created {args.samples} image-mask pairs in {args.output}")

