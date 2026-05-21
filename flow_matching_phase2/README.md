# Flow Matching Phase 2
Author : Tran Quoc Khanh - personal project
## Project Introduction

This project compares two class-conditional Flow Matching pipelines on CIFAR-10:

- `vanila_flow_matching` (baseline)
- `improve_flow_matching` (optimized and Kaggle-ready)

The goal is to understand why one setup produced noisy outputs while the improved setup produced class-recognizable images, and to keep the workflow reproducible for both local and Kaggle environments.

## Why This Project

We built this project for three practical reasons:

1. Improve generation quality under strict compute constraints (such as hardware, zero-shot model,etc)
2. Stabilize long training runs (up to hundreds of epochs) with safe resume (save checkpoint).
3. Add audit tools to detect whether generated samples are near-copies of training data, avoiding the risk of copy original images.

## Main Results

- Baseline training converged in loss but still produced noisy samples in many solver settings.
- Improved training produced much better semantic alignment per class (recognizable outputs across CIFAR-10 classes).
- Guidance sweep + NFE sweep gave controlled quality/performance trade-offs.
- Added Top-5 nearest-neighbor replicate check for memorization auditing.

## Real-World Applications

This workflow is useful for:

- Fast prototyping of class-conditional generative models.
- Edge-constrained diffusion/flow experiments with limited GPU budget.
- Educational demos for ODE samplers (Euler vs RK4) and CFG behavior.
- Safety audits for potential sample memorization.

## Workspace Structure

- `vanila_flow_matching/`: original baseline pipeline.
- `improve_flow_matching/`: optimized pipeline, Kaggle notebook, single-class inference script, replicate checker.
- `inference/`: checkpoints, logs, generated grids, replicate-check outputs.

## Run Baseline (Local)

```powershell
.\.venv\Scripts\python .\vanila_flow_matching\flow_matching_phase2_cifar10.py --config small --epochs 100 --batch 256
```

## Run Improved (Local)

```powershell
.\.venv\Scripts\python .\improve_flow_matching\Kaggle_improve_flow_matching.py --mode train --config auto --epochs 200 --batch 128
.\.venv\Scripts\python .\improve_flow_matching\Kaggle_improve_flow_matching.py --mode infer --checkpoint .\inference\improve_checkpoints\improve_best.pt
```

## Single-Class Inference (RK4)

Use:

- `improve_flow_matching/test_inference.py`

This script loads the trained improved checkpoint (default: `inference/improve_checkpoints/epoch_0500.pt`) and generates samples with RK4.

### Input options

- `--class_name` (recommended): text class input
- `--class_id`: numeric class label (`0..9`)

Valid class names:

- `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

### Example: frog generation

```powershell
.\.venv\Scripts\python .\improve_flow_matching\test_inference.py --class_name frog --samples 8 --rk4_steps 80 --guidance 2.0 --guidance_rescale 0.7
```

## Optional Replicate Check (Top-5 nearest)

```powershell
.\.venv\Scripts\python .\improve_flow_matching\test_inference.py --class_name frog --samples 8 --rk4_steps 80 --guidance 2.0 --guidance_rescale 0.7 --check_replicate --replicate_topk 5
```

Saved per generated image:

- `generated_<class>_<idx>.png`
- `top1_nearest_image<train_index>.png`
- `top2_nearest_image<train_index>.png`
- `top3_nearest_image<train_index>.png`
- `top4_nearest_image<train_index>.png`
- `top5_nearest_image<train_index>.png`
- `replicate_report.txt`

Output location:

- `inference/improve_inference/test_single_class/replicate_check_<class>_<prefix>/gen_<idx>/`

## Standalone Replicate Checker

```powershell
.\.venv\Scripts\python .\improve_flow_matching\check_replicate.py --generated_dir ..\inference\improve_inference\test_single_class --class_name frog --topk 5
```

Notes:

- The script first tries pretrained ResNet18 feature matching.
- If pretrained weights cannot be downloaded, it automatically falls back to pixel-space cosine matching.

## Kaggle

Use notebook:

- `improve_flow_matching/Kaggle_improve_flow_matching.ipynb`

## Cleanup Notes

- Heavy CIFAR extracted batch folder (`data/`) was removed.
- Critical checkpoints and inference outputs are preserved under `inference/`.
