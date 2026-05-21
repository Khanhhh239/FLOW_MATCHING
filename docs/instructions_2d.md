# Flow Matching: Practical Instructions & Commands

## Quick Start (30 seconds)

```powershell
# Run baseline with all metrics
python flow_matching.py --dataset 8gaussians --epochs 500

# Check results
cat outputs/metrics_8gaussians.json
```

---

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Command Reference](#command-reference)
5. [Troubleshooting](#troubleshooting)
6. [Output Files](#output-files)

---

## 1. Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup
```powershell
# Create conda environment
conda env create -f environment.yml
conda activate flow_matching

# Or install manually
pip install torch torchvision numpy matplotlib scipy
```

### Verify installation
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 2. Basic Usage

### 2.1 Baseline Model (Stable, Recommended for Learning)

**File:** `flow_matching.py`

**Features:**
- Stable on CPU and GPU
- 6 paper-level metrics
- 12-panel visualization
- Fast training

**Quick run:**
```powershell
python flow_matching.py --dataset 8gaussians --epochs 500
```

**Output:**
- `outputs/flow_matching_8gaussians.png` - Visualization
- `outputs/metrics_8gaussians.json` - All metrics

**Expected time:**
- CPU: ~5 minutes
- GPU (RTX 4050): ~30 seconds

### 2.2 Improved Model (Best Results)

**File:** `improve_flow_matching.py`

**Features:**
- EMA weights
- Time-dependent Ïƒ(t)
- Curvature regularization
- Rectified Flow (optional)
- NFE curves
- Comparison plots

**Quick run (without Rectified Flow):**
```powershell
python improve_flow_matching.py --dataset 8gaussians --epochs 300
```

**Best results (with Rectified Flow):**
```powershell
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --use_reflow --reflow_epochs 200
```

**Output:**
- `outputs/comparison_8gaussians.png` - Baseline vs Improved
- `outputs/improved_metrics_8gaussians.json` - Full comparison

**Expected time:**
- Without reflow: ~2-3 minutes (GPU)
- With reflow: ~5-8 minutes (GPU)

---

## 3. Advanced Usage

### 3.1 Swiss Roll Dataset

```powershell
# Baseline
python flow_matching.py --dataset swiss_roll --epochs 500

# Improved with reflow
python improve_flow_matching.py --dataset swiss_roll --epochs 400 --use_reflow --reflow_epochs 300
```

**Note:** Swiss Roll benefits from more epochs due to complex geometry.

### 3.2 Quick Testing (Fast Iteration)

```powershell
# Test on small dataset
python flow_matching.py --dataset 8gaussians --epochs 100 --n_train 10000

# Test improved version
python improve_flow_matching.py --dataset 8gaussians --epochs 50 --n_train 5000
```

**Use for:**
- Debugging
- Hyperparameter tuning
- Quick experiments

### 3.3 High-Quality Results (Publication)

```powershell
# Baseline (high quality)
python flow_matching.py `
    --dataset 8gaussians `
    --epochs 800 `
    --n_train 100000 `
    --batch 16384 `
    --hidden 512 `
    --depth 6

# Improved (best quality)
python improve_flow_matching.py `
    --dataset 8gaussians `
    --epochs 500 `
    --use_reflow `
    --reflow_epochs 400 `
    --n_train 100000 `
    --batch 16384 `
    --hidden 512 `
    --depth 5 `
    --curv_weight 0.02
```

**Expected results:**
- Straightness > 0.98
- Wâ‚‚ < 0.05
- MMD < 0.005
- Solver gap < 0.05

### 3.4 Ablation Studies

**Test individual improvements:**

```powershell
# Baseline
python improve_flow_matching.py --dataset 8gaussians --epochs 300

# + Curvature regularization only
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --curv_weight 0.05

# + Rectified Flow only
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --use_reflow --reflow_epochs 200 --curv_weight 0.0

# All improvements
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --use_reflow --reflow_epochs 200 --curv_weight 0.01
```

### 3.5 Experimental Features (Advanced Architecture)

```powershell
# Enable Fourier features + FiLM + Spectral norm
python improve_flow_matching.py `
    --dataset swiss_roll `
    --epochs 300 `
    --no-use_simple_arch `
    --use_spectral_norm `
    --use_reflow `
    --reflow_epochs 200
```

**Warning:** May be unstable on CPU. GPU recommended.

---

## 4. Command Reference

### 4.1 flow_matching.py (Baseline)

**Required arguments:**
- None (all have defaults)

**Common arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `8gaussians` | Dataset: `8gaussians` or `swiss_roll` |
| `--n_train` | `50000` | Number of training samples |
| `--epochs` | `500` | Training epochs |
| `--batch` | `8192` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--hidden` | `256` | Hidden dimension |
| `--depth` | `5` | Number of residual blocks |
| `--sigma_min` | `0.05` | Minimum noise floor |
| `--out_dir` | `outputs` | Output directory |

**Examples:**

```powershell
# Minimal (use defaults)
python flow_matching.py

# Custom dataset
python flow_matching.py --dataset swiss_roll

# Larger model
python flow_matching.py --hidden 512 --depth 6

# More training
python flow_matching.py --epochs 1000 --n_train 100000

# Faster testing
python flow_matching.py --epochs 100 --n_train 10000 --batch 4096
```

### 4.2 improve_flow_matching.py (Improved)

**Additional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_reflow` | `False` | Enable Rectified Flow (2-phase training) |
| `--reflow_epochs` | `200` | Epochs for reflow phase |
| `--use_ema` | `True` | Use EMA weights |
| `--ema_decay` | `0.999` | EMA decay rate |
| `--curv_weight` | `0.01` | Curvature regularization weight |
| `--use_simple_arch` | `True` | Use stable baseline architecture |
| `--use_spectral_norm` | `False` | Use spectral normalization |
| `--sigma_min` | `0.01` | Minimum noise floor |

**Examples:**

```powershell
# Minimal (EMA + time-dependent Ïƒ)
python improve_flow_matching.py --dataset 8gaussians --epochs 300

# Add curvature regularization
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --curv_weight 0.05

# Full improvements (recommended)
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --use_reflow --reflow_epochs 200 --curv_weight 0.01

# Disable EMA (not recommended)
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --no-use_ema

# Advanced architecture (experimental)
python improve_flow_matching.py --dataset swiss_roll --epochs 300 --no-use_simple_arch --use_spectral_norm
```

### 4.3 Hyperparameter Tuning Guide

**Learning rate:**
- Default: `3e-4` (good for most cases)
- Larger models: `1e-4` to `5e-4`
- Unstable training: `1e-4`
- Fast convergence: `5e-4` (may overfit)

**Batch size:**
- Default: `8192` (good balance)
- More GPU memory: `16384` (better mode coverage)
- Less GPU memory: `4096`
- CPU: `2048` to `4096`

**Model size:**
- Small (fast): `--hidden 128 --depth 3`
- Default: `--hidden 256 --depth 4-5`
- Large (best quality): `--hidden 512 --depth 6`
- Very large: `--hidden 1024 --depth 8` (may overfit on toy data)

**Curvature weight:**
- No regularization: `0.0`
- Light: `0.001` to `0.005`
- Default: `0.01`
- Strong: `0.05` to `0.1`
- Too strong (>0.1): May underfit

**Sigma min:**
- Very small: `0.001` (sharper, may be unstable)
- Default (improved): `0.01`
- Default (baseline): `0.05`
- Large: `0.1` (more stable, less sharp)

---

## 5. Troubleshooting

### 5.1 Common Issues

**Problem: NaN loss**

```powershell
# Solution 1: Reduce learning rate
python improve_flow_matching.py --lr 1e-4

# Solution 2: Increase sigma_min
python improve_flow_matching.py --sigma_min 0.05

# Solution 3: Disable curvature reg
python improve_flow_matching.py --curv_weight 0.0

# Solution 4: Use simple architecture (default)
python improve_flow_matching.py --use_simple_arch
```

**Problem: Mode collapse (coverage < 100%)**

```powershell
# Solution 1: Increase batch size
python flow_matching.py --batch 16384

# Solution 2: Train longer
python flow_matching.py --epochs 800

# Solution 3: Larger model
python flow_matching.py --hidden 512 --depth 6
```

**Problem: High Wâ‚‚ distance**

```powershell
# Solution 1: Train longer
python flow_matching.py --epochs 800

# Solution 2: Use improved version
python improve_flow_matching.py --epochs 300 --use_reflow --reflow_epochs 200

# Solution 3: Larger model
python flow_matching.py --hidden 512 --depth 6
```

**Problem: High solver gap**

```powershell
# Solution 1: Enable Rectified Flow (biggest impact)
python improve_flow_matching.py --use_reflow --reflow_epochs 200

# Solution 2: Increase curvature regularization
python improve_flow_matching.py --curv_weight 0.05

# Solution 3: Train longer
python improve_flow_matching.py --epochs 500
```

**Problem: Out of memory**

```powershell
# Solution 1: Reduce batch size
python flow_matching.py --batch 4096

# Solution 2: Smaller model
python flow_matching.py --hidden 128 --depth 3

# Solution 3: Fewer training samples
python flow_matching.py --n_train 25000
```

**Problem: Slow training**

```powershell
# Solution 1: Reduce model size
python flow_matching.py --hidden 128 --depth 3

# Solution 2: Fewer epochs
python flow_matching.py --epochs 300

# Solution 3: Smaller dataset
python flow_matching.py --n_train 25000

# Solution 4: Larger batch size (if GPU memory allows)
python flow_matching.py --batch 16384
```

### 5.2 Debugging Checklist

- [ ] Check GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Verify environment: `conda list | grep torch`
- [ ] Check output directory exists: `mkdir outputs`
- [ ] Try minimal example: `python flow_matching.py --epochs 10 --n_train 1000`
- [ ] Check for NaN in first epoch (if yes, reduce lr or increase sigma_min)
- [ ] Monitor GPU memory: `nvidia-smi` (if on GPU)

### 5.3 Performance Optimization

**For fastest training:**
```powershell
python flow_matching.py `
    --epochs 300 `
    --batch 16384 `
    --hidden 128 `
    --depth 3 `
    --n_train 25000
```

**For best quality:**
```powershell
python improve_flow_matching.py `
    --epochs 500 `
    --use_reflow `
    --reflow_epochs 400 `
    --batch 16384 `
    --hidden 512 `
    --depth 6 `
    --n_train 100000 `
    --curv_weight 0.02
```

**For balanced (recommended):**
```powershell
python improve_flow_matching.py `
    --epochs 300 `
    --use_reflow `
    --reflow_epochs 200 `
    --batch 8192 `
    --hidden 256 `
    --depth 4 `
    --curv_weight 0.01
```

---

## 6. Output Files

### 6.1 Baseline (flow_matching.py)

**Generated files:**
```
outputs/
â”œâ”€â”€ flow_matching_8gaussians.png      # 12-panel visualization
â””â”€â”€ metrics_8gaussians.json           # All metrics in JSON
```

**Visualization panels:**
1. Target distribution
2. Euler samples (100 steps)
3. RK4 samples (50 steps)
4. Training loss curve
5. ODE trajectories (24 particles)
6. Vector field at t=0.5
7. Density evolution
8. Mode coverage histogram
9. Euler vs RK4 overlay
10. Path efficiency distribution
11. Metrics summary

**JSON structure:**
```json
{
  "dataset": "8gaussians",
  "n_train": 50000,
  "epochs": 500,
  "final_loss": 1.234,
  "best_loss": 1.123,
  "straightness": 0.945,
  "wasserstein_w2": 0.314,
  "mmd": 0.002,
  "mode_coverage": 1.0,
  "path_efficiency": 0.635,
  "solver_gap": 2.513,
  "n_params": 698114,
  "device": "cuda",
  "torch_version": "2.5.1"
}
```

### 6.2 Improved (improve_flow_matching.py)

**Generated files:**
```
outputs/
â”œâ”€â”€ comparison_8gaussians.png          # 6-panel comparison
â””â”€â”€ improved_metrics_8gaussians.json   # Full comparison data
```

**Comparison panels:**
1. Quality metrics bar chart (Straightness, Efficiency, Coverage)
2. Distance metrics bar chart (Wâ‚‚, MMD, Solver Gap)
3. Kinetic energy comparison
4. NFE vs Wâ‚‚ curve
5. NFE vs MMD curve
6. Improvement summary

**JSON structure:**
```json
{
  "dataset": "8gaussians",
  "use_reflow": true,
  "baseline": {
    "straightness": 0.945,
    "wasserstein_w2": 0.314,
    "mmd": 0.002,
    ...
  },
  "improved": {
    "straightness": 0.975,
    "wasserstein_w2": 0.100,
    "mmd": 0.001,
    ...
  },
  "nfe_curve": [
    {"nfe": 10, "w2": 0.15, "mmd": 0.002},
    {"nfe": 20, "w2": 0.12, "mmd": 0.0015},
    ...
  ]
}
```

### 6.3 Reading Results

**Python:**
```python
import json

# Load metrics
with open('outputs/metrics_8gaussians.json') as f:
    metrics = json.load(f)

print(f"Straightness: {metrics['straightness']:.4f}")
print(f"W2 Distance: {metrics['wasserstein_w2']:.4f}")
print(f"Mode Coverage: {metrics['mode_coverage']*100:.1f}%")
```

**PowerShell:**
```powershell
# View metrics
cat outputs/metrics_8gaussians.json | ConvertFrom-Json | Format-List

# Extract specific metric
(cat outputs/metrics_8gaussians.json | ConvertFrom-Json).straightness
```

**Command line (jq):**
```bash
# If you have jq installed
jq '.straightness' outputs/metrics_8gaussians.json
jq '.wasserstein_w2' outputs/metrics_8gaussians.json
```

---

## 7. Batch Processing

### 7.1 Run Multiple Experiments

**PowerShell script:**
```powershell
# experiments.ps1

# Baseline
python flow_matching.py --dataset 8gaussians --epochs 500
python flow_matching.py --dataset swiss_roll --epochs 500

# Improved without reflow
python improve_flow_matching.py --dataset 8gaussians --epochs 300
python improve_flow_matching.py --dataset swiss_roll --epochs 300

# Improved with reflow
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --use_reflow --reflow_epochs 200
python improve_flow_matching.py --dataset swiss_roll --epochs 400 --use_reflow --reflow_epochs 300

Write-Host "All experiments completed!"
```

**Run:**
```powershell
.\experiments.ps1
```

### 7.2 Hyperparameter Sweep

```powershell
# sweep_curv_weight.ps1

$weights = @(0.0, 0.001, 0.005, 0.01, 0.05, 0.1)

foreach ($w in $weights) {
    Write-Host "Testing curvature weight: $w"
    python improve_flow_matching.py `
        --dataset 8gaussians `
        --epochs 200 `
        --curv_weight $w `
        --out_dir "outputs/sweep_curv_$w"
}

Write-Host "Sweep completed!"
```

### 7.3 Compare Results

```python
# compare_results.py
import json
import glob

results = []
for path in glob.glob('outputs/*/metrics_*.json'):
    with open(path) as f:
        data = json.load(f)
        results.append({
            'path': path,
            'straightness': data['straightness'],
            'w2': data['wasserstein_w2'],
            'solver_gap': data['solver_gap']
        })

# Sort by W2 distance
results.sort(key=lambda x: x['w2'])

print("Best results:")
for r in results[:5]:
    print(f"{r['path']}: W2={r['w2']:.4f}, Straightness={r['straightness']:.4f}")
```

---

## 8. Tips & Best Practices

### 8.1 For Learning

1. **Start simple:**
   ```powershell
   python flow_matching.py --epochs 100 --n_train 10000
   ```

2. **Visualize trajectories:**
   - Open `outputs/flow_matching_8gaussians.png`
   - Look at panel 5 (trajectories)
   - Should be nearly straight lines

3. **Check metrics:**
   - Straightness should be > 0.94
   - Mode coverage should be 100%
   - If not, train longer

4. **Experiment:**
   - Try different hyperparameters
   - Compare baseline vs improved
   - Understand what each improvement does

### 8.2 For Research

1. **Establish baseline:**
   ```powershell
   python flow_matching.py --dataset 8gaussians --epochs 500
   ```

2. **Test improvements individually:**
   - EMA only
   - Curvature reg only
   - Rectified Flow only
   - All combined

3. **Generate NFE curves:**
   ```powershell
   python improve_flow_matching.py --use_reflow --reflow_epochs 200
   ```

4. **Report all metrics:**
   - Wâ‚‚, MMD, Mode Coverage
   - Straightness, Path Efficiency
   - Solver Gap, Kinetic Energy

5. **Include visualizations:**
   - Trajectory plots
   - Comparison charts
   - NFE curves

### 8.3 For Production

1. **Train with best settings:**
   ```powershell
   python improve_flow_matching.py `
       --epochs 500 `
       --use_reflow `
       --reflow_epochs 400 `
       --hidden 512 `
       --depth 6
   ```

2. **Optimize NFE:**
   - Check NFE curve
   - Find minimum steps for acceptable quality
   - Use that for deployment

3. **Save model:**
   - Add model saving to code
   - Save EMA model (better quality)
   - Test on validation set

4. **Monitor metrics:**
   - Track Wâ‚‚ and MMD over time
   - Ensure mode coverage stays 100%
   - Watch for degradation

---

## 9. Quick Reference

### One-Line Commands

```powershell
# Fastest test
python flow_matching.py --epochs 50 --n_train 5000

# Standard baseline
python flow_matching.py --dataset 8gaussians --epochs 500

# Best results
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --use_reflow --reflow_epochs 200

# Swiss Roll
python flow_matching.py --dataset swiss_roll --epochs 500

# High quality
python improve_flow_matching.py --epochs 500 --use_reflow --reflow_epochs 400 --hidden 512 --depth 6
```

### Metric Thresholds

| Metric | Excellent | Good | Poor |
|--------|-----------|------|------|
| Straightness | >0.97 | 0.94-0.97 | <0.94 |
| Wass‚‚ Distance | <0.10 | 0.10-0.30 | >0.30 |
| MMD | <0.01 | 0.01-0.05 | >0.05 |
| Mode Coverage | 100% | 87.5% | <75% |
| Path Efficiency | >0.95 | 0.90-0.95 | <0.90 |
| Solver Gap | <0.05 | 0.05-0.15 | >0.15 |

### Expected Training Times (RTX 4050)

| Configuration | Time |
|---------------|------|
| Quick test (100 epochs) | ~30 sec |
| Baseline (500 epochs) | ~2 min |
| Improved (300 epochs) | ~2 min |
| Improved + Reflow (300+200) | ~5-8 min |
| High quality (500+400) | ~15 min |



