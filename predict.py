"""Run inference on one image and save its predicted binary mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model import TinyUNet


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TinyUNet(base_channels=checkpoint.get("base_channels", 16)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    original = Image.open(args.image).convert("RGB")
    resized = original.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0]
    mask = (probability >= args.threshold).cpu().numpy().astype(np.uint8) * 255
    output = Image.fromarray(mask).resize(original.size, Image.Resampling.NEAREST)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"Saved predicted mask to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_unet.pt")
    parser.add_argument("--output", default="outputs/predicted_mask.png")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())

