# analyze.md

## 1. Objective

This document analyzes why the baseline vanilla Flow Matching run produced heavily noisy samples, while the improved pipeline reached class-recognizable CIFAR-10 images.

## 2. Dataset Context (CIFAR-10)

The CIFAR-10 dataset is a foundational computer vision benchmark consisting of 60,000 low-resolution ($32 \times 32$ pixels) color images evenly distributed across 10 distinct classes of everyday objects and animals.

## 3. Baseline vs Improved - Technical Summary

### 3.1 Baseline (`vanila_flow_matching`)

Training Algo CFM:

Algorithm 1: Conditional Flow Matching Training

Input:
- Data distribution p_data(x1)
- Noise distribution p_noise(x0) = N(0, I)
- Neural vector field vθ(x, t)
- Number of training steps T

Repeat until convergence:
1. Sample clean data: x1 ~ p_data(x1)
2. Sample noise: x0 ~ p_noise(x0)
3. Sample time: t ~ Uniform(0, 1)
4. Construct interpolation path: xt = (1 - t)x0 + t x1
5. Define target velocity: ut = x1 - x0
6. Predict velocity with neural network: v_pred = vθ(xt, t)
7. Compute flow matching loss: L = ||v_pred - ut||²
8. Update parameters: θ ← θ - η ∇θ L

Inference / Sampling Algo CFM:

Algorithm 2: CFM Sampling (ODE Integration)

Input:
- Trained vector field vθ(x, t)
- Number of solver steps N
- Step size Δt = 1/N

1. Initialize: x0 ~ N(0, I)
2. For k = 0 ... N-1:
- tk = k / N
- Predict velocity: vk = vθ(xk, tk)
- Integrate ODE: xk+1 = xk + Δt · vk
3. Return final sample: xN

Core behavior:
- Precompute "Latents" from CIFAR-10, don't use VAE to convert images into latent vectors (because the dim of images is 32x32, quite small).
- If the selection in here is a high-dimension dataset (512x512 or 1024x1024 like ImageNet), we should apply a VAE to convert into latent spaces and use Flow Matching Algo in this spaces.
- Utilize Diffusion Transformer (DiT) to train (backbone), task is that input: noisy point x and semantic label c -> output: learn to predict that what noisy points need to remove in order to clean the image.
- Use CFM Loss (Conditional Flow Matching): it's simply the MSE between predicted velocity (from model DiT) and truth velocity (derivative of path).
- Standard CFG (classifier-free Guidance) merge at inference to combine conditions and unconditional branch.
- Stand out training and validating techniques:
  - Optimizer: AdamW
  - EMA (Exponential Moving Average)
  - Scheduler: OneCycleLR
  - AMP mixed precision
- Sampler (ODE):
  - Euler sampler: first-order approximation, global error = O(dt)
  - RK4 sampler: weighted 4-slope approximation, global error = O(dt^4), but NFE is 4x Euler steps
- Select model version small and train with around 100 epochs.

Observed issues:
- Loss diagram has extremely downward trend (a good signal).
- Samples stayed noisy for many solver settings (inference quality was very bad).
- Strong CFG often amplified artifacts.
- Visual class semantics were unstable for small/medium NFE.

### 3.2 Improved (`improve_flow_matching` + Kaggle notebook)

Added changes:
- Hardware & Scaling: Selected T4-friendly model configuration (MiniDiT-Base) and Gradient Accumulation (`grad_accum=2`) to effectively increase stable batch behavior under VRAM constraints.
- RAM-First Dataset Caching: Cached the entire preprocessed CIFAR-10 dataset into RAM to eliminate disk I/O bottlenecks during training.
- Dynamic Data Augmentation: Implemented on-the-fly horizontal flips via `torch.flip` inside `__getitem__` with `.clone()` protection.
- Attention Optimization: Integrated Native SDPA (`F.scaled_dot_product_attention`) with QK-Norm for long-horizon stability.
- Mixed Precision: Deployed AMP FP16 with dynamic GradScaler to maximize Tensor Core utilization and speed up training/inference.
- Advanced Flow Matching Dynamics: Shifted to Log-Normal Time Sampling to concentrate model capacity around critical mid-trajectory vector fields (`t ≈ 0.5`).
- Inference Guardrails: Implemented Guidance Rescale in CFG sampling to correct variance shifts and prevent over-saturation at high guidance.
- Fault-Tolerant Training: Resume-safe checkpointing, periodic backups, and OOM-safe recovery behavior.

Result:
- Train in Kaggle environment during ~6h40m.
- Much better semantic alignment per class in generated images.
- Lower tendency for speckle noise and color burn under guidance.
- Trial with guidance scales (1.0, 1.5, 2.0, 3.0, 4.0): low guidance = weak class conditioning, high guidance = stronger class fit but reduced creative diversity.
- More reliable long-run training behavior across many epochs.
- Resolution can improve further with more epochs and post-upscaling.

## 4. Mathematical Intuition Behind Quality Gap

### 4.1 Flow Matching objective

The model learns velocity field `u_theta(x_t, t, y)` toward a target field `u*` under MSE:

`L = E ||u_theta(x_t, t, y) - u*(x0, x1, t)||^2`.

In practice, generation quality depends on:
- field smoothness over `t`
- numerical integration error
- conditioning stability
- trajectory regions receiving enough learning signal

### 4.2 Why uniform time sampling can under-emphasize useful regions

With uniform `t`, training budget spreads equally over easy and hard trajectory segments. For image structure emergence, middle trajectory bands often carry high semantic transition signal. Log-normal-based mapping concentrates more samples around this useful region.

### 4.3 QK-Norm and long-horizon stability

Attention logits scale with `q·k/sqrt(d)`. Over long training, drift in q/k magnitude can destabilize softmax temperature. QK-Norm constrains query/key magnitudes before interaction and reduces variance explosions.

### 4.4 Guidance rescale and artifact suppression

Classic CFG:

`u_cfg = u_uncond + w (u_cond - u_uncond)`.

For large `w`, variance of `u_cfg` can overshoot, creating color burn/noisy high-frequency artifacts. Guidance rescale matches field scale back toward conditional statistics.

### 4.5 Solver/NFE interaction

Even with good model weights, low NFE introduces ODE discretization error. Baseline field quality + low NFE can yield severe grain. Improved field smoothness plus better guidance scaling allows clearer outputs at moderate NFE.

## 5. Replicate-Check Techniques (Top-5 Nearest Audit)

To test whether generated images are copied from training data, the improved pipeline uses Top-5 nearest-neighbor auditing.

Techniques used:
- Class-conditional nearest search: compare generated images only with train frog subset (same class).
- Feature-space cosine matching (preferred): ResNet18 penultimate embeddings + L2 normalization + cosine similarity.
- Fallback mode: if pretrained weights cannot be downloaded, use pixel-space cosine on flattened normalized tensors.
- Top-k retrieval (`k=5`) for each generated image.
- For each sample, save:
  - generated image
  - `top1_nearest_image<train_index>.png` ... `top5_nearest_image<train_index>.png`
  - `replicate_report.txt` : Note that Similar_Cosine ~ 0.98-1 is warning overfitting (Memorization), the ideal value is 0.6-0.85

This gives both quantitative similarity and direct visual inspection of potential memorization.

## 6. Training Efficiency Analysis

### 6.1 Vanilla bottlenecks

- Less optimized attention path.
- Weaker memory strategy around dataset handling.
- Less resilient behavior in long runs when resumed or interrupted.

### 6.2 Improved efficiency gains

- SDPA reduces attention overhead and improves GPU utilization.
- AMP halves activation memory footprint and increases matrix throughput.
- RAM caching removes repetitive storage bottlenecks.
- Structured checkpoint cadence lowers risk of wasted epochs on interruptions.

## 7. Why one model looked noisy and the improved one became recognizable

Primary reasons:
- Better optimization signal allocation in time (log-normal sampling).
- Better transformer numerical conditioning (QK-Norm).
- Better inference conditioning control (guidance rescale).
- Better runtime efficiency enabling stable longer training and richer evaluation sweeps.

Net effect: the improved model learns a smoother, more semantically faithful flow field, and the solver follows trajectories that preserve class-level structure instead of amplifying noise.

## 8. Practical Recommendations

- Keep baseline as reference for ablation only.
- Use improved pipeline for all serious training runs.
- Evaluate multiple guidance values and NFE settings (under `inference/improve_inference`).
- Preserve periodic checkpoints every fixed epoch window for safe continuation.
- Always run replicate-check auditing before claiming qualitative improvements.
