# Crack Semantic Segmentation

A compact, reproducible computer-vision project for detecting cracks at pixel level. It includes a small U-Net, a paired image/mask loader, Dice and IoU evaluation, inference, smoke tests, and a synthetic-data generator so the full pipeline can run without downloading a dataset.

This repository is a portfolio implementation: it demonstrates the workflow and engineering structure without claiming benchmark results that were not reproduced here.

## What is included

- `TinyUNet`: lightweight encoder-decoder segmentation model in PyTorch
- Paired image/mask dataset validation and preprocessing
- Deterministic train/validation split
- Class-weighted binary cross-entropy for sparse crack pixels
- Dice and Intersection over Union metrics
- Single-image inference script
- Synthetic crack generator for an immediate end-to-end run
- Pytest smoke tests
- GitHub Actions test workflow

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python generate_synthetic_data.py --samples 200
python train.py --epochs 10
python predict.py data/synthetic/images/sample_0000.png
python -m pytest -q
```

The trained checkpoint is written to `checkpoints/tiny_unet.pt`, and inference writes a mask to `outputs/predicted_mask.png`.

## Use a real dataset

Arrange paired files with identical names:

```text
data/my_dataset/
├── images/
│   ├── sample_001.png
│   └── sample_002.png
└── masks/
    ├── sample_001.png
    └── sample_002.png
```

Masks should be binary or grayscale images in which crack pixels are bright. Then run:

```bash
python train.py --data data/my_dataset --epochs 30
```

## Project structure

```text
.
├── generate_synthetic_data.py
├── predict.py
├── train.py
├── src/
│   ├── data.py
│   └── model.py
└── tests/
    └── test_smoke.py
```

## Possible extensions

- Add Albumentations for stronger geometric and photometric augmentation.
- Compare the baseline with a pretrained encoder or SegFormer.
- Track experiments with MLflow and export the best model to ONNX.
- Evaluate on a public crack dataset and report reproducible confidence intervals.

## License

MIT
