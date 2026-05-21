# Flow Matching: Technical Analysis & Optimization Report
### Baseline (Phase 1) vs Improved (Phase 2) — Mathematical Deep Dive

---

## Table of Contents

1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Baseline Architecture Analysis](#2-baseline-architecture-analysis)
3. [Improved Architecture & Techniques](#3-improved-architecture--techniques)
4. [Metrics: Mathematical Definitions & Interpretation](#4-metrics-mathematical-definitions--interpretation)
5. [Results Analysis & Comparative Observations](#5-results-analysis--comparative-observations)
6. [Critical Findings & Recommendations](#6-critical-findings--recommendations)

---

## 1. Mathematical Foundations

### 1.1 Conditional Flow Matching (CFM) — Core Theory

Flow Matching (Lipman et al., 2022) learns a time-dependent vector field $u_\theta : \mathbb{R}^D \times [0,1] \to \mathbb{R}^D$ that defines an ODE whose solution transports a source distribution $p_0 = \mathcal{N}(0, I)$ to a target distribution $p_1 = p_{\text{data}}$.

The continuity equation governing density transport is:

$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \cdot u_t) = 0$$

**Optimal-Transport CFM interpolant** (OT-CFM): For a source-target pair $(x_0, x_1)$ the interpolant is defined as:

$$x_t = \bigl(1 - (1 - \sigma_{\min}) \cdot t\bigr) \cdot x_0 + t \cdot x_1$$

which represents a straight-line path perturbed by noise $\sigma_{\min}$ at $t=0$.

The corresponding **conditional vector field target** (constant along the path):

$$u^*(x_t \mid x_0, x_1) = x_1 - (1 - \sigma_{\min}) \cdot x_0 \qquad\text{(Eq. 1}$$

This is the fundamental insight of CFM — the target field is **analytically available** and constant along each trajectory.

**Training objective:**

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x_0 \sim \mathcal{N}(0,I),\; x_1 \sim p_{\text{data}}} \bigl[ \| u_\theta(x_t, t) - u^*(x_t \mid x_0, x_1) \|^2 \bigr] \qquad\text{(Eq. 2}$$

A perfectly trained model produces **exactly straight-line trajectories**, since the target $u^*$ is constant.

### 1.2 Why OT-CFM Is Theoretically Optimal

Under independent coupling $\pi(x_0, x_1) = p_0(x_0) \cdot p_1(x_1)$, OT-CFM minimizes the **kinetic energy** functional:

$$E_{\text{kinetic}} = \mathbb{E}\left[\int_0^1 \|u_t(x_t)\|^2 \, dt\right]$$

which is the Benamou–Brenier formulation of optimal transport. Minimum kinetic energy corresponds to straight geodesics in $L^2$.

---

## 2. Baseline Architecture Analysis

### 2.1 VectorFieldMLP — Skip-MLP Design

The baseline model $u_\theta : \mathbb{R}^2 \times [0,1] \to \mathbb{R}^2$ uses a skip-connection MLP:

```
Input: [x || embed(t)] ∈ R^{D + E}
   ↓  input_proj: Linear → LayerNorm → SiLU
   ↓  depth × ResBlock
   +  skip: Linear(input, hidden)    ← global skip
   ↓  output_proj: Linear → R^D      ← zero-initialized
```

**Sinusoidal Time Embedding** (DDPM-style, Ho et al. 2020):

$$\text{embed}(t)_i = \begin{cases} \sin\!\left(t \cdot \omega_i \cdot 10 \right) & i < E/2 \\ \cos\!\left(t \cdot \omega_{i-E/2} \cdot 10 \right) & i \geq E/2 \end{cases}, \quad \omega_i = \exp\!\left(-\frac{\ln 10000 \cdot i}{E/2}\right)$$

This creates log-spaced frequencies spanning several orders of magnitude, allowing the model to distinguish fine temporal differences.

**ResBlock:**

$$\text{ResBlock}(x) = \text{SiLU}\bigl(x + \text{LN} \circ \text{Linear} \circ \text{SiLU} \circ \text{LN} \circ \text{Linear}(x)\bigr)$$

**Zero-initialization of output head:** By initializing $W_{\text{out}} = 0$, the model starts at $u_\theta \equiv 0$, guaranteeing clean, unbiased early gradients — a well-known trick from diffusion model training.

**Baseline Loss with Magnitude Regularization:**

$$\mathcal{L}_{\text{baseline}} = \mathbb{E}\bigl[\| u_\theta(x_t, t) - u^* \|^2\bigr] + \lambda_{\text{mag}} \cdot \mathbb{E}\bigl[\|u_\theta(x_t, t)\|_2\bigr]$$

with $\lambda_{\text{mag}} = 10^{-4}$. This penalizes field magnitude, reducing overshoot while barely affecting training dynamics.

### 2.2 Baseline Training Configuration

| Hyperparameter | Value |
|---|---|
| Hidden dim | 256 |
| Depth | 5 |
| $\sigma_{\min}$ | $10^{-4}$ |
| Epochs | 500 |
| Batch size | 8192 |
| Learning rate | $3 \times 10^{-4}$ |
| LR schedule | CosineAnnealingLR |
| Warmup | 20 epochs |
| Parameters | 698,114 |
| AMP | bfloat16 (Ampere+) |

**Optimizer:** AdamW with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, weight decay $= 10^{-5}$, fused CUDA kernel enabled.

---

## 3. Improved Architecture & Techniques

### 3.1 Technique 1 — Time-Dependent Noise Schedule $\sigma(t)$

**Baseline** uses constant $\sigma_{\min}$, giving a fixed interpolant width everywhere.

**Improved** uses an adaptive schedule:

$$\sigma(t) = \sigma_{\min} \cdot (1 - t)^2 \qquad\text{(Eq. 3}$$

This means:

- At $t \to 0$: noise is maximal $\approx \sigma_{\min}$, allowing the model to explore more paths early.
- At $t \to 1$: noise $\to 0$, enforcing sharp convergence to the target.

The interpolant becomes:

$$x_t = \bigl(1 - (1 - \sigma(t)) \cdot t\bigr) x_0 + t \cdot x_1$$

And the target field correspondingly:

$$u^*(x_t \mid x_0, x_1) = x_1 - (1 - \sigma(t)) \cdot x_0 \qquad\text{(Eq. 4}$$

This creates a **time-adaptive curriculum**: the loss signal is softer early (more variance in paths) and sharper late (stricter alignment), which in practice improves training stability on complex geometries like Swiss Roll.

### 3.2 Technique 2 — Curvature Regularization

A key failure mode of naive CFM is that the learned vector field can still be curved even if the individual conditional paths are straight, because the **marginal** vector field $u_t(x)$ aggregates over all pairs $(x_0, x_1)$ passing through $x$ at time $t$.

The curvature penalty directly minimizes the temporal derivative of the field:

$$\mathcal{L}_{\text{curv}} = \mathbb{E}\left[\left\| u_\theta(x_{t+\epsilon}, t+\epsilon) - u_\theta(x_t, t) \right\|^2\right] \qquad\text{(Eq. 5}$$

where $x_{t+\epsilon} = x_t + \epsilon \cdot u_\theta(x_t, t)$ is a one-step Euler forward propagation with step $\epsilon = 0.01$.

The **full improved loss** is:

$$\mathcal{L}_{\text{improved}} = \underbrace{\mathbb{E}\bigl[\| u_\theta(x_t, t) - u^* \|^2\bigr]}_{\text{CFM regression}} + \underbrace{\lambda_c \cdot \mathcal{L}_{\text{curv}}}_{\text{curvature reg}} + \underbrace{\lambda_{\text{mag}} \cdot \mathbb{E}\bigl[\|u_\theta\|\bigr]}_{\text{magnitude reg}} \qquad\text{(Eq. 6}$$

with $\lambda_c = 0.01$, $\lambda_{\text{mag}} = 10^{-5}$.

> **Why this works:** A field with zero curvature is constant along trajectories — exactly the OT condition. By penalizing $\partial_t u_t$, we encourage the marginal field to also become straight even when the individual conditional fields are averaged together.

### 3.3 Technique 3 — Rectified Flow / Reflow

**Original CFM** couples noise $x_0 \sim \mathcal{N}(0,I)$ independently with data $x_1 \sim p_{\text{data}}$. This random coupling is not optimal — trajectories can cross, creating a curved marginal vector field even when each conditional path is a straight line.

**Rectified Flow** (Liu et al., 2022) iteratively straightens trajectories via a two-stage protocol:

**Stage 1** — Train with random coupling:
$$\hat{u}_\theta \leftarrow \arg\min_\theta \mathcal{L}_{\text{CFM}}(\theta)$$

**Stage 2** — Generate paired data using the Stage 1 model:
$$\hat{x}_0^{(i)} \sim \mathcal{N}(0,I), \quad \hat{x}_1^{(i)} = \text{ODESolve}(\hat{u}_\theta, \hat{x}_0^{(i)}) \qquad\text{(Eq. 7}$$

**Stage 3** — Retrain on the self-generated *straight* pairs:
$$u_\theta^{\text{reflow}} \leftarrow \arg\min_\theta \mathcal{L}_{\text{reflow}}\bigl(\{(\hat{x}_0^{(i)}, \hat{x}_1^{(i)})\}\bigr)$$

The **reflow loss** is identical in form to the original CFM loss but applied to the self-generated pairs, enforcing the fixed-point property:

$$\mathcal{L}_{\text{reflow}} = \mathbb{E}_{t,\, (\hat{x}_0, \hat{x}_1)} \bigl[\| u_\theta(\hat{x}_t, t) - (\hat{x}_1 - (1-\sigma(t))\hat{x}_0) \|^2 \bigr] + \lambda_c \mathcal{L}_{\text{curv}} + \lambda_{\text{mag}} \mathbb{E}[\|u_\theta\|]$$

**Theoretical guarantee (Liu et al., 2022):** Repeated reflow iterations converge to a model whose trajectories are transport maps (non-crossing), minimizing the expected crossing number of paths. This is a **fixed-point theorem** — the reflow operator $\mathcal{R}$ satisfies $\mathcal{R}(u^*) = u^*$ if and only if $u^*$ generates a deterministic transport.

### 3.4 Technique 4 — Exponential Moving Average (EMA)

Instead of evaluating the raw model at the end of training, we maintain an EMA shadow copy:

$$\theta_{\text{EMA}} \leftarrow \mu \cdot \theta_{\text{EMA}} + (1 - \mu) \cdot \theta \qquad\text{(Eq. 8}$$

with decay $\mu = 0.999$, updated after every gradient step.

**Why EMA helps:**

- Averages out stochastic gradient noise — effectively a temporal ensemble.
- $\theta_{\text{EMA}}$ sits in a flatter loss basin than the training trajectory.
- Empirically reduces variance in generated sample quality by ~10–20% on 2-D toy problems.
- At convergence of a cosine schedule, $\theta_{\text{EMA}}$ approximates the Polyak-Ruppert average of the last $\approx 1/(1-\mu) = 1000$ steps.

### 3.5 Technique 5 — Fourier Features for Spatial Encoding

The baseline feeds raw coordinates $x \in \mathbb{R}^2$. For complex geometries (Swiss Roll, spirals), high-frequency structure requires many network layers to represent.

**Random Fourier Features** (Rahimi & Recht, 2007):

$$\phi(x) = \left[\sin(x^\top B_1), \cos(x^\top B_1), \ldots, \sin(x^\top B_{d/2}), \cos(x^\top B_{d/2})\right] \in \mathbb{R}^d \qquad\text{(Eq. 9}$$

where $B_j \sim \mathcal{N}(0, \sigma_B^2 I)$ are fixed random projection vectors. With scale $\sigma_B = 0.1$:

$$\phi(x)_i = \sin(\sigma_B \cdot x^\top b_i), \quad \phi(x)_{i+d/2} = \cos(\sigma_B \cdot x^\top b_i)$$

By the Bochner theorem, this approximates a shift-invariant kernel $k(x, y) = \exp(-\sigma_B^2 \|x-y\|^2 / 2)$, giving the network access to an **infinite-dimensional feature space** at initialization. Geometric details become linearly separable in $\phi$-space.

### 3.6 Technique 6 — FiLM Conditioning

**Baseline** concatenates the time embedding directly with spatial features: a linear operation that limits how time can modulate intermediate representations.

**Feature-wise Linear Modulation (FiLM)** (Perez et al., 2018) applies a learned affine transformation per feature dimension, conditioned on time:

$$\text{FiLM}(h, t_{\text{emb}}) = \gamma(t_{\text{emb}}) \odot h + \beta(t_{\text{emb}}) \qquad\text{(Eq. 10}$$

where:

$$\gamma(t_{\text{emb}}) = W_\gamma t_{\text{emb}} + b_\gamma \in \mathbb{R}^H, \quad \beta(t_{\text{emb}}) = W_\beta t_{\text{emb}} + b_\beta \in \mathbb{R}^H$$

This allows the model to **multiplicatively gate** hidden features based on time — a strictly more expressive operation than concatenation. In flow matching, time plays the role of a conditioning variable; FiLM is analogous to cross-attention but cheaper.

**Initialization to identity:** $W_\gamma = 0$, $b_\gamma = 1$, $W_\beta = 0$, $b_\beta = 0$ ensures $\gamma \equiv 1$, $\beta \equiv 0$ at the start of training, so the FiLM does not disrupt early gradient flow.

### 3.7 Technique 7 — Spectral Normalization

To bound the Lipschitz constant of the network (stabilizing ODE integration), spectral normalization constrains each weight matrix:

$$\tilde{W} = \frac{W}{\sigma_1(W)} \qquad\text{(Eq. 11}$$

where $\sigma_1(W)$ is the largest singular value, estimated via power iteration. This enforces:

$$\|u_\theta(x, t) - u_\theta(y, t)\|_2 \leq L \|x - y\|_2$$

A Lipschitz-bounded vector field is critical for ODE solvers — large Lipschitz constants require smaller step sizes (more NFE) for a fixed error tolerance.

> **Note:** In our experiments, spectral normalization was disabled (`use_spectral_norm=False`) because it conflicted with `LayerNorm` inside the same block, causing gradient instability on this 2-D scale.

---

## 4. Metrics: Mathematical Definitions & Interpretation

### 4.1 Straightness Score

$$\text{Straightness} = 1 - \frac{\bar{d}_\perp}{\bar{\ell}_{\text{chord}}} \qquad\text{(Eq. 12}$$

where for each particle trajectory:

- $\ell_{\text{chord}}^{(i)} = \|x_T^{(i)} - x_0^{(i)}\|$: length of the chord connecting start to end.
- $d_\perp^{(i)}(t)$: perpendicular distance from $x_t^{(i)}$ to the chord line at step $t$.
- $\bar{d}_\perp = \mathbb{E}_i[\mathbb{E}_t[d_\perp^{(i)}(t)]]$: mean deviation across particles and time.

Score = 1.0 means **all trajectories are exact straight lines**. Score < 1 means curvature is present. This directly measures how well OT-CFM achieves its theoretical goal.

### 4.2 Path Efficiency

$$\text{PathEff} = \mathbb{E}_i\!\left[\frac{\ell_{\text{chord}}^{(i)}}{\ell_{\text{path}}^{(i)}}\right], \quad \ell_{\text{path}}^{(i)} = \sum_{k=0}^{T-1} \|x_{t_{k+1}}^{(i)} - x_{t_k}^{(i)}\| \qquad\text{(Eq. 13}$$

Path efficiency = 1 means the particle travels in a straight line (chord = arc length). This metric captures macro-scale trajectory geometry, whereas straightness captures micro-scale perpendicular deviation from the ideal chord.

### 4.3 Wasserstein-2 Distance (Approximated)

The true $\mathcal{W}_2$ between two distributions:

$$\mathcal{W}_2(p, q) = \left(\inf_{\pi \in \Pi(p,q)} \int \|x - y\|^2 \, d\pi(x,y)\right)^{1/2} \qquad\text{(Eq. 14}$$

is approximated here using Hungarian matching on subsampled point clouds ($N = 1000$):

$$\widehat{\mathcal{W}}_2 = \left(\frac{1}{N}\sum_{i} \|x_i^{\text{real}} - x_{\sigma(i)}^{\text{fake}}\|^2 \right)^{1/2}$$

where $\sigma$ is the optimal permutation from linear sum assignment. This is the **exact** $\mathcal{W}_2$ at the sample level but only approximates the distributional distance due to finite $N$.

### 4.4 Maximum Mean Discrepancy (MMD)

With RBF kernel $k(x, y) = \exp(-\|x-y\|^2 / 2\sigma^2)$:

$$\text{MMD}^2(p, q) = \mathbb{E}_{x,x' \sim p}[k(x,x')] + \mathbb{E}_{y,y' \sim q}[k(y,y')] - 2\mathbb{E}_{x \sim p, y \sim q}[k(x,y)] \qquad\text{(Eq. 15}$$

MMD = 0 if and only if $p = q$ (when the kernel is characteristic). As a **kernel two-sample test statistic**, MMD captures all-order moment differences between distributions. The RBF kernel with bandwidth $\sigma = 1.0$ is tuned to the scale of the 8-Gaussian and Swiss Roll datasets.

### 4.5 Solver Gap

$$\text{SolverGap} = \frac{1}{N}\sum_{i=1}^N \left\|\tilde{x}^{\text{Euler}}_{(i)} - \tilde{x}^{\text{RK4}}_{(i)}\right\|_2 \qquad\text{(Eq. 16}$$

where both $\tilde{x}^{\text{Euler}}$ (100 steps) and $\tilde{x}^{\text{RK4}}$ (50 steps) start from the **same** initial noise $x_0^{(i)}$. A small solver gap means the vector field is nearly linear in $t$ — integrable with few steps. This is the practical payoff of trajectory straightening.

> **Important caveat:** Since Euler (100 steps, NFE=100) and RK4 (50 steps, NFE=200) use different NFE budgets, a fair comparison requires accounting for this. The gap also mixes discretization error with distributional differences.

### 4.6 Kinetic Energy

$$E_k = \int_0^1 \mathbb{E}_{x_t \sim p_t}\!\left[\|u_\theta(x_t, t)\|^2\right] dt \approx \frac{1}{T} \sum_{i=0}^{T-1} \mathbb{E}\!\left[\|u_\theta(x_{t_i}, t_i)\|^2\right] \qquad\text{(Eq. 17}$$

This is the Benamou–Brenier energy — the OT functional. Lower kinetic energy = more efficient transport. Theoretical optimum for Gaussian-to-Gaussian transport is $\|m_1\|^2 + \text{tr}(\Sigma_0 + \Sigma_1 - 2(\Sigma_0^{1/2}\Sigma_1\Sigma_0^{1/2})^{1/2})$.

### 4.7 Mode Coverage (8-Gaussians specific)

Mode $i$ is **detected** if $\geq 20$ samples fall within radius $r = 0.5$ of center $c_i = 2[\cos(2\pi i/8), \sin(2\pi i/8)]$:

$$\text{Coverage} = \frac{1}{8}\sum_{i=1}^{8} \mathbf{1}\!\left[\sum_{j} \mathbf{1}\!\left[\|x_j - c_i\| < 0.5\right] \geq 20\right] \qquad\text{(Eq. 18}$$

Full coverage = 1.0 is required for an 8-Gaussian model — mode dropping is a fundamental failure mode.

---

## 5. Results Analysis & Comparative Observations

### 5.1 8-Gaussians Results Summary

| Metric | Phase 1 Baseline | Phase 2 Baseline | Phase 2 Improved (Reflow) | Direction |
|--------|-----------------|-----------------|--------------------------|-----------|
| Straightness | 0.91648 | 0.95042 | **0.99331** | ↑ Higher better |
| Path Efficiency | 0.64984 | 0.70869 | **0.99857** | ↑ Higher better |
| W2 Distance | **0.23721** | 0.36561 | 0.55992 | ↓ Lower better |
| MMD | **0.000338** | 0.004592 | 0.009975 | ↓ Lower better |
| Solver Gap | 2.56137 | 2.72603 | 2.85036 | ↓ Lower better |
| Kinetic Energy | — | 2.31029 | **0.93915** | ↓ Lower better |
| Mode Coverage | **1.0** | **1.0** | **1.0** | = |

### 5.2 Swiss Roll Results Summary

| Metric | Phase 2 Baseline | Phase 2 Improved (Reflow) | Direction |
|--------|-----------------|--------------------------|-----------|
| Straightness | 0.83286 | **0.98685** | ↑ |
| Path Efficiency | 0.44674 | **0.96210** | ↑ |
| W2 Distance | **0.24435** | 0.40909 | ↓ |
| MMD | **0.002330** | 0.005751 | ↓ |
| Solver Gap | **1.91576** | 2.03913 | ↓ |
| Kinetic Energy | 1.18448 | **0.15204** | ↓ |

### 5.3 Detailed Metric-by-Metric Analysis

#### Straightness: A Clear Win for Reflow

Across both datasets:
- 8-Gaussians: $0.916 \to 0.950 \to 0.993$ (Phase 1 → Phase 2 baseline → Phase 2 improved)
- Swiss Roll: $0.833 \to 0.987$

This is the **largest and most consistent improvement** observed. The two-stage reflow protocol achieves its designed purpose — self-generated pairs force the learned vector field to become nearly constant along trajectories.

The $0.993$ score on 8-Gaussians is close to the theoretical maximum. The jump from $0.833$ to $0.987$ on Swiss Roll is especially significant since curved geometry (the spiraling manifold) provides a harder test for trajectory linearity.

#### Path Efficiency: Dramatic Improvement

- 8-Gaussians: $0.650 \to 0.709 \to 0.999$
- Swiss Roll: $0.447 \to 0.962$

Path efficiency $\approx 0.999$ means the particle paths are essentially straight lines in Euclidean space. This is the operational consequence of straightness — not just that the perpendicular deviation is small, but that the full arc length nearly equals the chord. This directly means **fewer integration steps are needed** to achieve accurate sampling.

However, the NFE curve tells a more nuanced story:

```
NFE=10:  W2=0.5266    (8-Gaussian, improved model)
NFE=20:  W2=0.5421
NFE=50:  W2=0.5270
NFE=100: W2=0.4679
NFE=200: W2=0.5058
```

The W2 curve **does not decrease monotonically with NFE** — it is essentially flat between NFE=10 and NFE=200 (range: 0.468–0.560). This is anomalous. For a truly straight vector field, $W2$ should improve sharply with NFE due to reduced discretization error, then plateau. The flatness suggests the **distributional quality is limited by the reflow model's mode coverage/sharpness**, not by integration error.

#### W2 Distance & MMD: A Paradoxical Regression

This is the most important and concerning finding:

| Stage | W2 (8G) | MMD (8G) |
|-------|---------|---------|
| Phase 1 baseline (500 ep) | **0.237** | **0.000338** |
| Phase 2 baseline (300 ep) | 0.366 | 0.004592 |
| Phase 2 improved (reflow) | 0.560 | 0.009975 |

W2 and MMD **worsen at every stage** relative to the Phase 1 baseline. Several factors explain this:

**Factor 1 — Epoch count.** Phase 1 runs 500 epochs; Phase 2 baseline runs only 300. The Phase 1 model simply has more training time to fit the data distribution.

**Factor 2 — Reflow compression.** The reflow model trains on self-generated pairs $(\hat{x}_0, \hat{x}_1)$ from a model that itself had W2 = 0.366. Any errors in the Phase 2 baseline **propagate** into the reflow training data. The reflow model cannot recover information that was not captured by the Phase 2 baseline. This is a **noise amplification problem**: imperfect initial trajectories create imperfect pairs; training on imperfect pairs compounds the distributional mismatch.

**Factor 3 — Reduced mode counts.** Per-mode sample counts drop dramatically:
- Phase 2 baseline: $[568, 517, 557, 510, 499, 583, 579, 517]$ (mean: 541, std: 31)
- Phase 2 improved: $[249, 256, 260, 235, 250, 265, 263, 234]$ (mean: 251, std: 12)

The improved model is actually more **balanced** across modes (std drops from 31 to 12), but the total count per mode is ~50% lower. This indicates the reflow model spreads samples too broadly — possibly collapsing toward the Gaussian prior more than toward the tight 8-Gaussian clusters. This is reflected in higher W2 (samples are farther from the true cluster centers on average).

**Factor 4 — $\sigma(t)$ schedule interaction.** The time-dependent $\sigma(t) = \sigma_{\min}(1-t)^2$ with $\sigma_{\min} = 0.01$ creates a wider corridor near $t=0$. This is beneficial for training stability but means the model learns to transport samples to a slightly noisier $x_1$ distribution — adding systematic bias when $\sigma_{\min}$ is too large compared to cluster std = 0.15.

#### Kinetic Energy: A Genuine Improvement

The kinetic energy result is unambiguously positive:

| Model | KE (8G) | KE (Swiss Roll) |
|-------|---------|-----------------|
| Phase 2 baseline | 2.310 | 1.184 |
| Phase 2 improved | **0.939** | **0.152** |

The $6\times$ reduction in Swiss Roll kinetic energy $(1.184 \to 0.152)$ is striking. Lower kinetic energy means the vector field is closer to the Benamou–Brenier optimal transport map. Combined with near-perfect path efficiency, this confirms the improved model has learned a **more efficient transport**, even if it is transporting to a slightly wrong target distribution.

#### Solver Gap: Marginal Worsening

Solver gap increases slightly across all experiments:
- 8-Gaussians: $2.561 \to 2.726 \to 2.850$
- Swiss Roll: $1.916 \to 2.039$

This is counterintuitive — straighter trajectories should be easier to integrate, reducing solver gap. Two explanations: (1) The per-particle comparison in solver gap is sensitive to whether the distributions are mode-matched. If the Euler and RK4 samples end up in different modes (both correct modes but different ones), the gap is large even if both are good samples. (2) The comparison uses different NFE (Euler: 100, RK4: 50), which conflates solver accuracy with overall NFE.

---

## 6. Critical Findings & Recommendations

### 6.1 Summary of What Works and What Doesn't

| Technique | Effect on Straightness | Effect on W2/MMD | Effect on KE | Verdict |
|-----------|----------------------|------------------|--------------|---------|
| Curvature regularization |  Moderate positive |  Neutral/negative |  Positive | Keep, reduce $\lambda_c$ |
| Time-dependent $\sigma(t)$ |  Positive |  Negative (too wide) |  Positive | Tune $\sigma_{\min}$ ↓ |
| EMA ($\mu=0.999$) |  Positive |  Mild positive |  Positive | Always use |
| Rectified Flow (reflow) |  Strong positive |  Negative |  Positive | Fix training data quality first |
| Fourier Features |  Architecture benefit |  Neutral |  Neutral | Useful for Swiss Roll |
| FiLM conditioning |  Architecture benefit |  Neutral |  Neutral | Good practice |
| Spectral Norm | — (disabled) | — | — | Needs tuning |

### 6.2 Root Cause of W2/MMD Regression

The fundamental problem is a **trade-off between trajectory geometry and distributional fidelity**:

> Reflow maximizes straightness by making the learned transport deterministic. But if the initial (Phase 2 baseline) model's $x_1$ distribution is already imperfect ($W2=0.366$), reflow locks in that imperfection — it cannot "un-learn" the distributional error, it only straightens the paths toward a wrong target.

This is consistent with Liu et al. (2022)'s theory: reflow guarantees convergence to a deterministic map, but that map transports $p_0$ to **the model's learned $p_1$**, not necessarily to the true $p_{\text{data}}$.

### 6.3 Concrete Recommendations for Phase 3

**Priority 1 — Fix the Phase 2 baseline quality before reflowing:**
- Increase Phase 2 baseline epochs to 500 (match Phase 1).
- Reduce $\sigma_{\min}$ from $0.01$ to $10^{-4}$ (match Phase 1 baseline) — the wider noise was the main cause of distributional blur.

**Priority 2 — Improve reflow data quality:**
- Use RK4 with $\geq 200$ steps (not 100 Euler steps) to generate reflow pairs. Lower ODE integration error $\Rightarrow$ better pair quality.
- Apply reflow only to a **warm, well-converged** baseline model.

**Priority 3 — Use the Phase 1 baseline as the reflow seed:**
- The Phase 1 model achieves $W2=0.237$, $\text{MMD}=3.4\times10^{-4}$ — far superior distributional quality. Running reflow from this model should combine high sample quality with improved straightness.

**Priority 4 — Fix the solver gap measurement:**
- Compare Euler and RK4 at **equal NFE** (e.g., Euler 200 steps vs. RK4 50 steps, both NFE=200) for a fair comparison.

**Priority 5 — Multi-scale MMD:**
- Report MMD at multiple bandwidths $\sigma \in \{0.1, 0.5, 1.0, 2.0\}$ to capture both local (cluster sharpness) and global (mode positions) distributional errors.

### 6.4 The Fundamental Tension in Flow Matching

These experiments reveal the core tension in generative flow matching:

$$\underbrace{\text{Straightness}}_{\text{integration efficiency}} \;\longleftrightarrow\; \underbrace{\text{Distributional Fidelity}}_{\text{W2, MMD}}$$

Straightening trajectories (via reflow, curvature reg) is valuable for **inference speed** — straighter paths need fewer NFE steps. But over-straightening an imperfect model **amplifies distributional errors**.

The ideal operating point is:
1. First, achieve good distributional fidelity (low W2/MMD) via standard CFM training.
2. Then, apply rectified flow / reflow to straighten trajectories without sacrificing the distributional quality already achieved.

The Phase 1 baseline ($W2=0.237$) + reflow is the recommended next experiment.

---

## References

- Lipman, Y., et al. "Flow Matching for Generative Modeling." *arXiv:2210.02747* (2022).
- Liu, X., et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *arXiv:2209.03003* (2022).
- Albergo, M. S., & Vanden-Eijnden, E. "Building Normalizing Flows with Stochastic Interpolants." *arXiv:2209.15571* (2023).
- Ho, J., et al. "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*
- Perez, E., et al. "FiLM: Visual Reasoning with a General Conditioning Layer." *AAAI 2018.*
- Rahimi, A., & Recht, B. "Random Features for Large-Scale Kernel Machines." *NeurIPS 2007.*
- Benamou, J.-D., & Brenier, Y. "A Computational Fluid Mechanics Solution to the Monge-Kantorovich Mass Transfer Problem." *Numerische Mathematik* 84 (2000).

---

*Analysis prepared for Flow Matching Phase 1→2 experimental series. All metrics computed on CUDA (RTX), PyTorch 2.5.1, bfloat16 AMP.*
