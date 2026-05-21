# Flow Matching (CFM) Project

This repository implements **Continuous Normalizing Flows (Flow Matching)** in PyTorch. It is divided into two main phases, evolving from simple 2D distributions to complex image generation on CIFAR-10.

## Why This Project Exists
This project bridges the gap between theoretical papers and practical implementations. It provides:
1. Clear, reproducible baselines for 2D distributions and CIFAR-10 image generation.
2. Quantitative evaluations with paper-level metrics (W2, MMD, Straightness, NFE) rather than just visual inspection.
3. Stabilized long training runs and safety audits to detect memorization (copying original training images).

## Project Structure

```text
cfm-pytorch/
├── README.md                 ← This project overview + quick-start
├── environment.yml           ← Conda environment configuration
├── .gitignore                ← Ignored checkpoints and cache files
├── notebooks/                
│   └── 01_kaggle_image_flow_matching.ipynb  ← Kaggle notebook for CIFAR-10
├── src/                      ← Extracted source code
│   ├── 2d_flow/              ← Phase 1: 2D Flow (8 Gaussians, Swiss Roll)
│   │   ├── vanilla_flow.py   ← Basic Flow Matching
│   │   └── improved_flow.py  ← Improved (RK45, weight decay, reflow)
│   └── image_flow/           ← Phase 2: Image Flow (CIFAR-10)
│       ├── vanilla_cifar10.py ← Latents, training, and inference pipeline
│       ├── test_inference.py  ← Inference script with CFG
│       └── check_replicate.py ← Auditing script to check for memorization
├── docs/                     ← Detailed mathematical analysis
│   ├── 2d_flow_analysis.md   
│   └── image_flow_analysis.md
├── assets/                   ← Media for documentation
└── results/                  ← Saved outputs (Ignored by Git)
    ├── 2d_flow_outputs/      ← Phase 1 plots and JSON metrics
    └── image_flow_inference/ ← Phase 2 checkpoints, grids, latents
```

---

## Quick Start & Commands

### 1. Environment Setup

```powershell
conda env create -f environment.yml
conda activate flow_matching
```

### 2. Phase 1: 2D Flow Matching

**Baseline Model:** Stable, fast, 6 paper-level metrics, and 12-panel visualization.
```powershell
# Minimal test run (30 seconds on GPU)
python src/2d_flow/vanilla_flow.py --dataset 8gaussians --epochs 500
```

**Improved Model:** Adds EMA, curvature regularization, time-dependent $\sigma$, and Rectified Flow.
```powershell
# High quality with 2-phase Rectified Flow
python src/2d_flow/improved_flow.py --dataset 8gaussians --epochs 300 --use_reflow --reflow_epochs 200
```
*Outputs (visualizations and `.json` metrics) are automatically saved to `results/2d_flow_outputs/`.*

### 3. Phase 2: Image Flow Matching (CIFAR-10)

**Train Baseline (Local):**
```powershell
# This command runs all 3 steps: Precompute latents -> Train -> Infer
python src/image_flow/vanilla_cifar10.py --mode all --config small --epochs 50
```

**Single-Class Inference with Classifier-Free Guidance (CFG):**
Generates images using a trained checkpoint (default loads from `results/image_flow_inference/improve_checkpoints/epoch_0500.pt`).
```powershell
python src/image_flow/test_inference.py --class_name frog --samples 8 --rk4_steps 80 --guidance 2.0 --guidance_rescale 0.7
```

**Replicate Check (Memorization Audit):**
Generates images and compares them against the training dataset using a Top-5 nearest neighbor search to ensure the model isn't just copying data.
```powershell
python src/image_flow/test_inference.py --class_name frog --samples 8 --check_replicate --replicate_topk 5
```
*Generated grids, individual images, nearest neighbors, and text reports are saved in `results/image_flow_inference/improve_inference/test_single_class/`.*

---

## Advanced Usage & Hyperparameters

### 2D Flow Options
- `--dataset`: `8gaussians` or `swiss_roll` (Note: Swiss roll needs more epochs, e.g., `--epochs 500`).
- `--hidden 512 --depth 6`: Increase model capacity for highly complex distributions.
- `--use_reflow --reflow_epochs 400`: Enable 2-phase Rectified Flow for straighter trajectories (drastically improves solver gap).

### Image Flow Options
- `--config`: `tiny` (3.5M params, suitable for CPU testing), `small` (14M, good for local RTX GPUs), `base` (33M, best for RTX 3080/4090 or Kaggle).
- `--guidance`: CFG scale (e.g., `2.0` is recommended).

---

## Main Results & Highlights

- **Phase 1 (2D):** Successfully matches complex 2D distributions. Upgrading to RK45 sampler drastically reduces ODE integration error and improves Earth Mover's Distance (EMD). The baseline runs provide a flawless playground to understand Flow Matching mathematically.
- **Phase 2 (Images):** Implements a Flow-Matching Diffusion Transformer (FM-DiT) for CIFAR-10. Improved training achieves recognizable semantic alignment per class. 
- **Audit Tools:** The integrated top-5 nearest-neighbor replicate check effectively prevents zero-shot models from blindly memorizing training images, keeping your workflow safe and robust.

Please refer to `docs/2d_flow_analysis.md` and `docs/image_flow_analysis.md` for deep-dives into the mathematical evaluations.
