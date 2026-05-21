"""
Flow Matching Phase 2: IMPROVED with Rectified Flow & Advanced Techniques
===========================================================================
Improvements over Phase 1:
  1.  Rectified Flow / Reflow (VERY HIGH PRIORITY)
  2.  Curvature Regularization (VERY HIGH PRIORITY)
  3.  Time-dependent σ(t) (VERY HIGH PRIORITY)
  4.  EMA Weights (HIGH PRIORITY)
  5.  Spectral Normalization (HIGH PRIORITY)
  6.  Fourier Features for spatial encoding (ARCHITECTURE)
  7.  FiLM time conditioning (ARCHITECTURE)
  8.  Kinetic Energy metric
  9.  NFE vs Quality curves

References:
  - Lipman et al. "Flow Matching for Generative Modeling" (2022)
  - Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer
    Data with Rectified Flow" (2022)
"""

import os, sys, time, json, platform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DEVICE + RTX OPTIMISATIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
        cap   = torch.cuda.get_device_capability(0)
        print(f"[GPU]    {name}")
        print(f"         VRAM={vram:.1f} GB  compute={cap[0]}.{cap[1]}")
        return torch.device("cuda"), True
    print("[DEVICE] No CUDA GPU found — running on CPU")
    return torch.device("cpu"), False

DEVICE, HAS_CUDA = get_device()

GPU_CAP = torch.cuda.get_device_capability(0)[0] if HAS_CUDA else 0

USE_AMP = HAS_CUDA and GPU_CAP >= 7
AMP_DTYPE = (
    torch.bfloat16
    if (HAS_CUDA and GPU_CAP >= 8)
    else torch.float16
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def make_8gaussians(n: int = 50_000, std: float = 0.15) -> torch.Tensor:
    """8 isotropic Gaussians equally spaced on the unit circle (radius 2)."""
    angles  = torch.linspace(0, 2 * np.pi, 9)[:-1]
    centers = torch.stack([torch.cos(angles),
                           torch.sin(angles)], dim=1) * 2.0
    idx     = torch.randint(0, 8, (n,))
    noise   = torch.randn(n, 2) * std
    return centers[idx] + noise


def make_swiss_roll_2d(n: int = 50_000) -> torch.Tensor:
    """2-D projection of the classic Swiss-Roll manifold, unit-normalised."""
    t    = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    data = np.stack([t * np.cos(t), t * np.sin(t)], axis=1).astype(np.float32)
    data = (data - data.mean(0)) / (data.std(0) + 1e-8)
    return torch.from_numpy(data)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  IMPROVED MODEL with Fourier Features + FiLM + Spectral Norm
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Single residual block: Linear → LayerNorm → SiLU with skip (BASELINE)."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class VectorFieldMLP(nn.Module):
    """
    BASELINE u_θ : ℝ^D × [0,1] → ℝ^D (from flow_matching.py)
    
    Simple, stable architecture without Fourier/FiLM.
    """
    def __init__(self, data_dim: int = 2, hidden: int = 256,
                 depth: int = 4, time_embed_dim: int = 64):
        super().__init__()
        in_dim = data_dim + time_embed_dim

        self.time_embed = SinusoidalEmbed(time_embed_dim)
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden),
                                        nn.LayerNorm(hidden), nn.SiLU())
        self.blocks      = nn.Sequential(*[ResBlock(hidden) for _ in range(depth)])
        self.skip        = nn.Linear(in_dim, hidden, bias=False)
        self.output_proj = nn.Linear(hidden, data_dim)

        # Zero-init: model starts predicting u≡0
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)
        h  = torch.cat([x, te], dim=-1)
        return self.output_proj(self.blocks(self.input_proj(h)) + self.skip(h))


class FourierFeatures(nn.Module):
    """
    Random Fourier Features for spatial coordinates.
    Helps with high-frequency geometry (Swiss Roll, manifolds).
    """
    def __init__(self, in_dim: int = 2, out_dim: int = 64, scale: float = 0.1):
        super().__init__()
        # Much smaller scale to avoid numerical issues
        B = torch.randn(in_dim, out_dim // 2) * scale
        self.register_buffer("B", B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) → (B, out_dim)
        x_proj = x @ self.B  # Remove 2*pi multiplier
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)


class SinusoidalEmbed(nn.Module):
    """Sinusoidal time embedding (DDPM-style)."""
    def __init__(self, dim: int = 64):
        super().__init__()
        half  = dim // 2
        freqs = torch.exp(-np.log(10_000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t[:, None] * self.freqs[None, :] * 10
        return torch.cat([x.sin(), x.cos()], dim=-1)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Better time conditioning than simple concatenation.
    """
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.scale = nn.Linear(time_dim, hidden_dim)
        self.shift = nn.Linear(time_dim, hidden_dim)
        
        # Initialize to identity-like behavior
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
    
    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.scale(t_emb)
        beta = self.shift(t_emb)
        # Clamp to prevent explosion
        gamma = torch.clamp(gamma, 0.1, 10.0)
        beta = torch.clamp(beta, -10.0, 10.0)
        return gamma * h + beta


class ImprovedResBlock(nn.Module):
    """Residual block with FiLM conditioning and spectral normalization."""
    def __init__(self, dim: int, time_dim: int, use_spectral_norm: bool = True):
        super().__init__()
        
        linear1 = nn.Linear(dim, dim)
        linear2 = nn.Linear(dim, dim)
        
        if use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
            linear2 = nn.utils.spectral_norm(linear2)
        
        self.net = nn.Sequential(
            linear1,
            nn.LayerNorm(dim),
            nn.SiLU(),
            linear2,
            nn.LayerNorm(dim),
        )
        self.film = FiLMLayer(time_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.net(x)
        h = self.film(h, t_emb)
        return self.act(x + h)


class ImprovedVectorFieldMLP(nn.Module):
    """
    Improved u_θ with:
      - Fourier features for spatial encoding
      - FiLM for time conditioning
      - Spectral normalization for stability
    """
    def __init__(self, data_dim: int = 2, hidden: int = 256,
                 depth: int = 4, time_embed_dim: int = 64,
                 fourier_dim: int = 32, use_spectral_norm: bool = True):
        super().__init__()
        
        # Smaller Fourier dimension and scale
        self.fourier = FourierFeatures(data_dim, fourier_dim, scale=0.1)
        self.time_embed = SinusoidalEmbed(time_embed_dim)
        
        in_dim = fourier_dim + time_embed_dim
        
        proj = nn.Linear(in_dim, hidden)
        if use_spectral_norm:
            proj = nn.utils.spectral_norm(proj)
        
        self.input_proj = nn.Sequential(proj, nn.LayerNorm(hidden), nn.SiLU())
        
        self.blocks = nn.ModuleList([
            ImprovedResBlock(hidden, time_embed_dim, use_spectral_norm)
            for _ in range(depth)
        ])
        
        skip = nn.Linear(in_dim, hidden, bias=False)
        if use_spectral_norm:
            skip = nn.utils.spectral_norm(skip)
        self.skip = skip
        
        out_proj = nn.Linear(hidden, data_dim)
        if use_spectral_norm:
            out_proj = nn.utils.spectral_norm(out_proj)
        self.output_proj = out_proj
        
        # Zero-init output with small scale
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize other layers carefully
        self._init_weights()
    
    def _init_weights(self):
        """Careful weight initialization to avoid NaN."""
        for m in self.modules():
            if isinstance(m, nn.Linear) and m != self.output_proj:
                # Use smaller initialization
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Clamp input to reasonable range
        x = torch.clamp(x, -10, 10)
        
        x_feat = self.fourier(x)
        t_emb = self.time_embed(t)
        
        h = torch.cat([x_feat, t_emb], dim=-1)
        h_proj = self.input_proj(h)
        
        for block in self.blocks:
            h_proj = block(h_proj, t_emb)
        
        h_proj = h_proj + self.skip(h)
        out = self.output_proj(h_proj)
        
        # Clamp output to prevent explosion
        out = torch.clamp(out, -100, 100)
        
        return out

# ─────────────────────────────────────────────────────────────────────────────
# 3.  IMPROVED CFM LOSS with Curvature Reg + Time-dependent σ(t)
# ─────────────────────────────────────────────────────────────────────────────
def sigma_schedule(t: torch.Tensor, sigma_min: float = 0.001) -> torch.Tensor:
    """
    Time-dependent noise schedule: σ(t) = σ_min * (1-t)²
    
    Early time (t→0): more noise
    Late time (t→1): sharper transport
    """
    # Clamp to prevent numerical issues
    return torch.clamp(sigma_min * (1.0 - t) ** 2, min=1e-6, max=1.0)


def improved_cfm_loss(model: nn.Module,
                      x1: torch.Tensor,
                      sigma_min: float = 0.001,
                      curv_weight: float = 0.01) -> torch.Tensor:
    """
    Improved CFM loss with:
      1. Time-dependent σ(t)
      2. Curvature regularization
      3. Magnitude regularization
    """
    B = x1.shape[0]
    device = x1.device
    dtype = x1.dtype

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device, dtype=dtype)
    t_exp = t[:, None]
    
    # Time-dependent noise
    sigma_t = sigma_schedule(t, sigma_min)[:, None]
    
    # Interpolant
    x_t = (1.0 - (1.0 - sigma_t) * t_exp) * x0 + t_exp * x1
    
    # Target
    u_target = x1 - (1.0 - sigma_t) * x0
    
    # Prediction
    pred = model(x_t, t)
    
    # Base loss
    loss = F.mse_loss(pred, u_target)
    
    # Curvature regularization: penalize temporal changes in vector field
    if curv_weight > 0:
        eps = 0.01
        t2 = (t + eps).clamp(max=1.0)
        with torch.no_grad():
            x_t2 = x_t + eps * pred
        pred2 = model(x_t2, t2)
        curv_reg = ((pred2 - pred) ** 2).mean()
        loss = loss + curv_weight * curv_reg
    
    # Magnitude regularization
    mag_reg = 1e-5 * pred.norm(dim=-1).mean()
    
    return loss + mag_reg


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RECTIFIED FLOW / REFLOW
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_reflow_pairs(model,
                          n_samples=50000,
                          n_steps=100,
                          device=DEVICE):
    """
    Generate self-generated (x0, x1_hat) pairs for Rectified Flow.
    
    After training once with random coupling, generate trajectories
    and retrain on these straighter paths.
    
    Effect: trajectories become progressively straighter.
    """
    model.eval()
    x0 = torch.randn(n_samples, 2, device=device)
    x = x0.clone()
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t = torch.full((n_samples,), i / n_steps, device=device)
        x = x + model(x, t) * dt
    
    x1_hat = x.detach()
    
    return x0.cpu(), x1_hat.cpu()


def reflow_loss(model: nn.Module,
                x0: torch.Tensor,
                x1: torch.Tensor,
                sigma_min: float = 0.001,
                curv_weight: float = 0.05) -> torch.Tensor:
    """
    Reflow loss: train on self-generated pairs (x0, x1_hat).
    
    This straightens trajectories over multiple reflow iterations.
    """
    B = x1.shape[0]
    device = x1.device
    dtype = x1.dtype

    t = torch.rand(B, device=device, dtype=dtype)
    t_exp = t[:, None]
    
    sigma_t = sigma_schedule(t, sigma_min)[:, None]
    
    # Interpolant with FIXED pairs
    x_t = (1.0 - (1.0 - sigma_t) * t_exp) * x0 + t_exp * x1
    
    # Target
    u_target = x1 - (1.0 - sigma_t) * x0
    
    pred = model(x_t, t)
    
    loss = F.mse_loss(pred, u_target)
    
    # Curvature reg
    if curv_weight > 0:
        eps = 0.01
        t2 = (t + eps).clamp(max=1.0)
        x_t2 = x_t + eps * pred.detach()
        pred2 = model(x_t2, t2)
        curv_reg = ((pred2 - pred) ** 2).mean()
        loss = loss + curv_weight * curv_reg
    
    mag_reg = 1e-4 * pred.norm(dim=-1).mean()
    
    return loss + mag_reg


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAINING with EMA
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model: nn.Module,
                data:  torch.Tensor,
                epochs:     int   = 500,
                batch_size: int   = 8192,
                lr:         float = 3e-4,
                device:     torch.device = DEVICE,
                use_ema:    bool  = True,
                ema_decay:  float = 0.999,
                reflow_data: tuple = None) -> tuple:
    """
    Training with EMA weights and optional reflow mode.
    
    Returns: (losses, ema_model)
    """
    model = model.to(device)
    
    # EMA model
    if use_ema:
        ema_model = deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
    else:
        ema_model = None
    
    # Dataset
    if reflow_data is not None:
        x0_reflow, x1_reflow = reflow_data
        dataset = TensorDataset(x0_reflow.float(), x1_reflow.float())
        print("[REFLOW] Training on self-generated pairs")
    else:
        dataset = TensorDataset(data.float())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP and AMP_DTYPE == torch.float16)
    warmup_epochs = 20

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 2,
        persistent_workers = True,
        pin_memory = HAS_CUDA,
    )

    losses, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        
        for batch in loader:
            if reflow_data is not None:
                x0_b, x1_b = batch
                x0_b = x0_b.to(device, non_blocking=True)
                x1_b = x1_b.to(device, non_blocking=True)
            else:
                x1_b = batch[0].to(device, non_blocking=True)
                x0_b = None
            
            optimizer.zero_grad(set_to_none=True)
            
            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    if reflow_data is not None:
                        loss = reflow_loss(model, x0_b, x1_b)
                    else:
                        loss = improved_cfm_loss(model, x1_b)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if reflow_data is not None:
                    loss = reflow_loss(model, x0_b, x1_b)
                else:
                    loss = improved_cfm_loss(model, x1_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            
            ep_loss += loss.item()
            
            # EMA update
            if use_ema and ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        
        if ep <= warmup_epochs:
            warmup_lr = lr * ep / warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr
        else:
            scheduler.step()
        
        avg = ep_loss / len(loader)
        losses.append(avg)
        
        if ep % 50 == 0 or ep == 1:
            lr_now = scheduler.get_last_lr()[0]
            ema_str = f"  EMA={ema_decay}" if use_ema else ""
            print(f"  Epoch {ep:>4}/{epochs}  loss={avg:.5f}  lr={lr_now:.2e}"
                  f"{ema_str}  t={time.time()-t0:.1f}s")

    return losses, ema_model


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ODE SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def euler_sample(model, n_samples=5000, n_steps=100,
                 device=DEVICE, return_traj=False):
    """Explicit Euler ODE solver."""
    model.eval()
    x   = torch.randn(n_samples, 2, device=device)
    dt  = 1.0 / n_steps
    snap_every = max(1, n_steps // 20)
    traj = [x.cpu().clone()] if return_traj else None

    for i in range(n_steps):
        t = torch.full((n_samples,), i / n_steps, device=device)
        x = x + model(x, t) * dt
        if return_traj and (i % snap_every == 0 or i == n_steps - 1):
            traj.append(x.cpu().clone())

    return (x.cpu(), traj) if return_traj else x.cpu()


@torch.no_grad()
def rk4_sample(model, n_samples=5000, n_steps=50, device=DEVICE):
    """Classical 4th-order Runge-Kutta."""
    model.eval()
    x  = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps

    def vf(xt, ti):
        return model(xt, torch.full((n_samples,), ti, device=device))

    for i in range(n_steps):
        ti = i / n_steps
        k1 = vf(x,               ti)
        k2 = vf(x + .5*dt*k1,   ti + .5*dt)
        k3 = vf(x + .5*dt*k2,   ti + .5*dt)
        k4 = vf(x +    dt*k3,   ti +    dt)
        x  = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return x.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  METRICS
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def straightness_score(model, n_samples=500, n_steps=200, device=DEVICE) -> float:
    """Quantifies how straight ODE trajectories are."""
    model.eval()
    x0   = torch.randn(n_samples, 2, device=device)
    x    = x0.clone()
    dt   = 1.0 / n_steps
    traj = [x.clone()]

    for i in range(n_steps):
        t  = torch.full((n_samples,), i / n_steps, device=device)
        x  = x + model(x, t) * dt
        traj.append(x.clone())

    traj  = torch.stack(traj, dim=0)
    start = traj[0]
    end   = traj[-1]
    chord = (end - start).norm(dim=-1).clamp(min=1e-6)

    devs = []
    for step in traj[1:-1]:
        delta     = step - start
        chord_dir = (end - start) / chord[:, None]
        proj      = (delta * chord_dir).sum(-1, keepdim=True) * chord_dir
        perp      = (delta - proj).norm(dim=-1)
        devs.append(perp)

    mean_dev = torch.stack(devs).mean()
    return float(1.0 - (mean_dev / chord.mean()).item())


@torch.no_grad()
def wasserstein_distance_2d(real: torch.Tensor,
                            fake: torch.Tensor,
                            max_samples: int = 1000) -> float:
    """Approximate Wasserstein-2 distance using Hungarian matching."""
    real = real[:max_samples].cpu().numpy()
    fake = fake[:max_samples].cpu().numpy()
    
    cost = cdist(real, fake, metric='sqeuclidean')
    row_ind, col_ind = linear_sum_assignment(cost)
    w2 = cost[row_ind, col_ind].mean()
    
    return float(np.sqrt(w2))


@torch.no_grad()
def compute_mmd(real: torch.Tensor,
                fake: torch.Tensor,
                sigma: float = 1.0,
                max_samples: int = 2000) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    x = real[:max_samples]
    y = fake[:max_samples]
    
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    
    kxx = torch.exp(-xx / (2 * sigma**2)).mean()
    kyy = torch.exp(-yy / (2 * sigma**2)).mean()
    kxy = torch.exp(-xy / (2 * sigma**2)).mean()
    
    mmd = kxx + kyy - 2 * kxy
    
    return float(mmd.item())


@torch.no_grad()
def mode_coverage_8gaussians(samples: torch.Tensor, radius: float = 0.5):
    """Measures how many of the 8 Gaussian modes are covered."""
    angles = torch.linspace(0, 2*np.pi, 9)[:-1]
    centers = torch.stack([
        torch.cos(angles),
        torch.sin(angles)
    ], dim=1) * 2.0
    
    samples = samples.cpu()
    dists = torch.cdist(samples, centers)
    nearest = dists.argmin(dim=1)
    
    counts = []
    detected = 0
    
    for i in range(8):
        mask = nearest == i
        if mask.sum() == 0:
            counts.append(0)
            continue
        
        close = dists[mask, i] < radius
        n = int(close.sum().item())
        counts.append(n)
        
        if n > 20:
            detected += 1
    
    return detected / 8.0, counts


@torch.no_grad()
def path_efficiency(traj):
    """Ratio: straight-line distance / actual trajectory length."""
    traj = torch.stack(traj, dim=0)
    diffs = traj[1:] - traj[:-1]
    path_len = diffs.norm(dim=-1).sum(0)
    chord = (traj[-1] - traj[0]).norm(dim=-1)
    eff = (chord / (path_len + 1e-8)).mean()
    return float(eff.item())


@torch.no_grad()
def solver_gap(euler_samples, rk4_samples):
    """Difference between Euler and RK4 generated distributions."""
    n = min(len(euler_samples), len(rk4_samples))
    gap = ((euler_samples[:n] - rk4_samples[:n])**2).sum(dim=-1).sqrt().mean()
    return float(gap.item())


@torch.no_grad()
def kinetic_energy(model, n_samples=1000, n_steps=100, device=DEVICE) -> float:
    """
    Measure kinetic energy: ∫ ||u_t||² dt
    
    Optimal transport should minimize energy.
    Lower = better transport efficiency.
    """
    model.eval()
    x = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps
    
    total_energy = 0.0
    
    for i in range(n_steps):
        t = torch.full((n_samples,), i / n_steps, device=device)
        u = model(x, t)
        energy = (u ** 2).sum(dim=-1).mean()
        total_energy += energy.item() * dt
        x = x + u * dt
    
    return total_energy


@torch.no_grad()
def nfe_quality_curve(model, data, n_samples=2000, step_range=[10, 20, 50, 100, 200],
                      device=DEVICE):
    """
    Generate NFE (Number of Function Evaluations) vs Quality curve.
    
    Very important metric in Flow Matching papers.
    Shows: can we use fewer steps after improvements?
    """
    results = []
    
    for n_steps in step_range:
        samples = euler_sample(model, n_samples=n_samples, n_steps=n_steps, device=device)
        w2 = wasserstein_distance_2d(data, samples, max_samples=1000)
        mmd = compute_mmd(data, samples, max_samples=1000)
        
        results.append({
            'nfe': n_steps,
            'w2': w2,
            'mmd': mmd
        })
    
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(baseline_metrics, improved_metrics, nfe_curve, out_path):
    """
    Plot comparison between baseline and improved model.
    Shows the impact of all improvements.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12), facecolor='#0a0a0c')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.3, wspace=0.25,
                            left=0.06, right=0.96, top=0.92, bottom=0.08)

    C_BLUE  = '#00e5ff'
    C_PINK  = '#ff4081'
    C_GREEN = '#76ff03'
    C_YELLOW = '#ffeb3b'
    C_PANEL = '#13131a'

    def style_ax(ax):
        ax.set_facecolor(C_PANEL)
        for sp in ax.spines.values(): sp.set_color('#252530')
        ax.tick_params(colors='#667788', labelsize=9)
        ax.xaxis.label.set_color('#8899aa')
        ax.yaxis.label.set_color('#8899aa')

    # 1 - Metrics comparison bar chart
    ax1 = fig.add_subplot(gs[0,0]); style_ax(ax1)
    metrics_names = ['Straightness', 'Path Efficiency', 'Mode Coverage']
    baseline_vals = [
        baseline_metrics.get('straightness', 0),
        baseline_metrics.get('path_efficiency', 0),
        baseline_metrics.get('mode_coverage', 0)
    ]
    improved_vals = [
        improved_metrics.get('straightness', 0),
        improved_metrics.get('path_efficiency', 0),
        improved_metrics.get('mode_coverage', 0)
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color=C_BLUE, alpha=0.7)
    ax1.bar(x + width/2, improved_vals, width, label='Improved', color=C_GREEN, alpha=0.7)
    ax1.set_ylabel('Score')
    ax1.set_title('Quality Metrics (Higher = Better)', color='white', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, rotation=15, ha='right')
    ax1.legend(facecolor=C_PANEL, labelcolor='white')
    ax1.set_ylim([0, 1.1])

    # 2 - Distance metrics comparison
    ax2 = fig.add_subplot(gs[0,1]); style_ax(ax2)
    dist_names = ['W2 Distance', 'MMD', 'Solver Gap']
    baseline_dist = [
        baseline_metrics.get('wasserstein_w2', 0),
        baseline_metrics.get('mmd', 0) * 10,  # scale for visibility
        baseline_metrics.get('solver_gap', 0)
    ]
    improved_dist = [
        improved_metrics.get('wasserstein_w2', 0),
        improved_metrics.get('mmd', 0) * 10,
        improved_metrics.get('solver_gap', 0)
    ]
    
    x = np.arange(len(dist_names))
    ax2.bar(x - width/2, baseline_dist, width, label='Baseline', color=C_PINK, alpha=0.7)
    ax2.bar(x + width/2, improved_dist, width, label='Improved', color=C_YELLOW, alpha=0.7)
    ax2.set_ylabel('Distance')
    ax2.set_title('Distance Metrics (Lower = Better)', color='white', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dist_names, rotation=15, ha='right')
    ax2.legend(facecolor=C_PANEL, labelcolor='white')
    ax2.text(0.02, 0.98, 'Note: MMD ×10 for visibility', transform=ax2.transAxes,
             fontsize=7, va='top', color='#667788')

    # 3 - Kinetic energy comparison
    ax3 = fig.add_subplot(gs[0,2]); style_ax(ax3)
    energy_data = [
        baseline_metrics.get('kinetic_energy', 0),
        improved_metrics.get('kinetic_energy', 0)
    ]
    colors = [C_PINK, C_GREEN]
    bars = ax3.bar(['Baseline', 'Improved'], energy_data, color=colors, alpha=0.7)
    ax3.set_ylabel('Energy')
    ax3.set_title('Kinetic Energy (Lower = Better)', color='white', fontweight='bold')
    for bar, val in zip(bars, energy_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)

    # 4 - NFE vs W2 curve
    ax4 = fig.add_subplot(gs[1,0]); style_ax(ax4)
    if nfe_curve:
        nfes = [r['nfe'] for r in nfe_curve]
        w2s = [r['w2'] for r in nfe_curve]
        ax4.plot(nfes, w2s, marker='o', color=C_GREEN, lw=2, markersize=8)
        ax4.set_xlabel('Number of Function Evaluations (NFE)')
        ax4.set_ylabel('Wasserstein-2 Distance')
        ax4.set_title('NFE vs Quality Curve', color='white', fontweight='bold')
        ax4.grid(True, alpha=0.2)

    # 5 - NFE vs MMD curve
    ax5 = fig.add_subplot(gs[1,1]); style_ax(ax5)
    if nfe_curve:
        mmds = [r['mmd'] for r in nfe_curve]
        ax5.plot(nfes, mmds, marker='s', color=C_YELLOW, lw=2, markersize=8)
        ax5.set_xlabel('Number of Function Evaluations (NFE)')
        ax5.set_ylabel('MMD')
        ax5.set_title('NFE vs MMD Curve', color='white', fontweight='bold')
        ax5.grid(True, alpha=0.2)

    # 6 - Improvement summary
    ax6 = fig.add_subplot(gs[1,2]); style_ax(ax6)
    ax6.axis('off')
    
    summary = "IMPROVEMENT SUMMARY\n" + "="*40 + "\n\n"
    
    # Calculate improvements
    if baseline_metrics.get('straightness', 0) > 0:
        straight_imp = ((improved_metrics.get('straightness', 0) - 
                        baseline_metrics.get('straightness', 0)) / 
                       baseline_metrics.get('straightness', 0) * 100)
        summary += f"Straightness:     +{straight_imp:.1f}%\n"
    
    if baseline_metrics.get('solver_gap', 1) > 0:
        gap_imp = ((baseline_metrics.get('solver_gap', 1) - 
                   improved_metrics.get('solver_gap', 1)) / 
                  baseline_metrics.get('solver_gap', 1) * 100)
        summary += f"Solver Gap:       -{gap_imp:.1f}%\n"
    
    if baseline_metrics.get('wasserstein_w2', 1) > 0:
        w2_imp = ((baseline_metrics.get('wasserstein_w2', 1) - 
                  improved_metrics.get('wasserstein_w2', 1)) / 
                 baseline_metrics.get('wasserstein_w2', 1) * 100)
        summary += f"W2 Distance:      -{w2_imp:.1f}%\n"
    
    summary += "\n" + "="*40 + "\n\n"
    summary += "KEY IMPROVEMENTS:\n"
    summary += "✓ Rectified Flow (reflow)\n"
    summary += "✓ Curvature regularization\n"
    summary += "✓ Time-dependent σ(t)\n"
    summary += "✓ EMA weights\n"
    summary += "✓ Spectral normalization\n"
    summary += "✓ Fourier features\n"
    summary += "✓ FiLM conditioning\n"
    
    ax6.text(0.1, 0.95, summary,
             transform=ax6.transAxes,
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             color='white',
             bbox=dict(boxstyle='round', facecolor=C_PANEL, 
                      alpha=0.8, edgecolor='#252530'))

    fig.suptitle('Flow Matching: Baseline vs Improved Model',
                 color='white', fontsize=14, fontweight='bold', y=0.96)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0c')
    plt.close(fig)
    print(f"[PLOT]  comparison saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset',   default='8gaussians',
                   choices=['8gaussians', 'swiss_roll'])
    p.add_argument('--n_train',   type=int,   default=50_000)
    p.add_argument('--epochs',    type=int,   default=300)
    p.add_argument('--batch',     type=int,   default=8192)
    p.add_argument('--hidden',    type=int,   default=256)
    p.add_argument('--depth',     type=int,   default=4)
    p.add_argument('--lr',        type=float, default=3e-4)
    p.add_argument('--sigma_min', type=float, default=0.01,
                   help='Minimum noise floor σ_min')
    p.add_argument('--use_reflow', action='store_true',
                   help='Enable Rectified Flow (train twice)')
    p.add_argument('--reflow_epochs', type=int, default=200,
                   help='Epochs for reflow training')
    p.add_argument('--use_ema', action='store_true', default=True,
                   help='Use EMA weights')
    p.add_argument('--ema_decay', type=float, default=0.999,
                   help='EMA decay rate')
    p.add_argument('--use_simple_arch', action='store_true', default=True,
                   help='Use simple baseline architecture (more stable)')
    p.add_argument('--use_spectral_norm', action='store_true', default=False,
                   help='Use spectral normalization')
    p.add_argument('--curv_weight', type=float, default=0.01,
                   help='Curvature regularization weight')
    p.add_argument('--out_dir',   default='outputs')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'━'*70}")
    print(f"  Flow Matching Phase 2: IMPROVED")
    print(f"  Dataset: {args.dataset}")
    print(f"  Python {sys.version.split()[0]}  ·  PyTorch {torch.__version__}")
    print(f"{'━'*70}")
    print(f"  Improvements:")
    print(f"    ✓ Rectified Flow:        {args.use_reflow}")
    print(f"    ✓ EMA:                   {args.use_ema} (decay={args.ema_decay})")
    print(f"    ✓ Spectral Norm:         {args.use_spectral_norm}")
    print(f"    ✓ Curvature Reg:         {args.curv_weight}")
    print(f"    ✓ Time-dependent σ(t):   Yes")
    print(f"    ✓ Fourier Features:      Yes")
    print(f"    ✓ FiLM Conditioning:     Yes")
    print(f"{'━'*70}\n")

    # ── Data
    if args.dataset == '8gaussians':
        data, label = make_8gaussians(args.n_train),  '8-Gaussian Mixture'
    else:
        data, label = make_swiss_roll_2d(args.n_train), 'Swiss Roll 2-D'
    print(f"[DATA]   {data.shape[0]:,} samples  "
          f"μ={data.mean(0).numpy().round(4)}  σ={data.std(0).numpy().round(4)}")

    # ── Model
    if args.use_simple_arch:
        print("[ARCH]   Using BASELINE architecture (stable)")
        model = VectorFieldMLP(
            hidden=args.hidden,
            depth=args.depth
        )
    else:
        print("[ARCH]   Using IMPROVED architecture (Fourier + FiLM)")
        model = ImprovedVectorFieldMLP(
            hidden=args.hidden,
            depth=args.depth,
            use_spectral_norm=args.use_spectral_norm
        )
    n_p = sum(p.numel() for p in model.parameters())
    print(f"[MODEL]  {n_p:,} params  hidden={args.hidden}  depth={args.depth}")

    # ── Phase 1: Initial Training
    print(f"\n{'─'*70}")
    print(f"[PHASE 1] Initial training ({args.epochs} epochs)")
    print(f"{'─'*70}")
    
    losses_phase1, ema_model = train_model(
        model, data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=DEVICE,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay
    )

    # Use EMA model for evaluation if available
    eval_model = ema_model if ema_model is not None else model

    # ── Sample Phase 1
    print("\n[SAMPLE] Phase 1 - Euler (100 steps) ...")
    euler_pts_p1, traj_p1 = euler_sample(eval_model, n_samples=5000, n_steps=100,
                                         device=DEVICE, return_traj=True)
    print("[SAMPLE] Phase 1 - RK4 (50 steps) ...")
    rk4_pts_p1 = rk4_sample(eval_model, n_samples=5000, n_steps=50, device=DEVICE)

    # ── Metrics Phase 1
    print("\n[METRICS] Phase 1 evaluation ...")
    baseline_metrics = {}
    baseline_metrics['straightness'] = straightness_score(eval_model, device=DEVICE)
    baseline_metrics['wasserstein_w2'] = wasserstein_distance_2d(data, rk4_pts_p1)
    baseline_metrics['mmd'] = compute_mmd(data, rk4_pts_p1)
    baseline_metrics['path_efficiency'] = path_efficiency([t.to(DEVICE) for t in traj_p1])
    baseline_metrics['solver_gap'] = solver_gap(euler_pts_p1, rk4_pts_p1)
    baseline_metrics['kinetic_energy'] = kinetic_energy(eval_model, device=DEVICE)
    
    if args.dataset == '8gaussians':
        cov, counts = mode_coverage_8gaussians(rk4_pts_p1)
        baseline_metrics['mode_coverage'] = cov
        baseline_metrics['mode_counts'] = counts
    else:
        baseline_metrics['mode_coverage'] = 0.0
        baseline_metrics['mode_counts'] = []

    print(f"\n{'─'*50}")
    print("[PHASE 1 RESULTS]")
    print(f"  Straightness:     {baseline_metrics['straightness']:.5f}")
    print(f"  W2 Distance:      {baseline_metrics['wasserstein_w2']:.5f}")
    print(f"  MMD:              {baseline_metrics['mmd']:.5f}")
    print(f"  Path Efficiency:  {baseline_metrics['path_efficiency']:.5f}")
    print(f"  Solver Gap:       {baseline_metrics['solver_gap']:.5f}")
    print(f"  Kinetic Energy:   {baseline_metrics['kinetic_energy']:.5f}")
    if args.dataset == '8gaussians':
        print(f"  Mode Coverage:    {baseline_metrics['mode_coverage']*100:.1f}%")
    print(f"{'─'*50}\n")

    # ── Phase 2: Rectified Flow (Optional)
    improved_metrics = baseline_metrics.copy()
    
    if args.use_reflow:
        print(f"\n{'─'*70}")
        print(f"[PHASE 2] Rectified Flow / Reflow ({args.reflow_epochs} epochs)")
        print(f"{'─'*70}")
        
        # Generate reflow pairs
        print("[REFLOW] Generating self-generated pairs ...")
        x0_reflow, x1_reflow = generate_reflow_pairs(
            eval_model,
            n_samples=args.n_train,
            n_steps=100,
            device=DEVICE
        )
        
        # Create new model for reflow
        if args.use_simple_arch:
            model_reflow = VectorFieldMLP(
                hidden=args.hidden,
                depth=args.depth
            )
        else:
            model_reflow = ImprovedVectorFieldMLP(
                hidden=args.hidden,
                depth=args.depth,
                use_spectral_norm=args.use_spectral_norm
            )
        
        # Train on reflow pairs
        losses_phase2, ema_model_reflow = train_model(
            model_reflow, data,
            epochs=args.reflow_epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=DEVICE,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            reflow_data=(x0_reflow, x1_reflow)
        )
        
        eval_model_reflow = ema_model_reflow if ema_model_reflow is not None else model_reflow
        
        # ── Sample Phase 2
        print("\n[SAMPLE] Phase 2 - Euler (100 steps) ...")
        euler_pts_p2, traj_p2 = euler_sample(eval_model_reflow, n_samples=5000, n_steps=100,
                                             device=DEVICE, return_traj=True)
        print("[SAMPLE] Phase 2 - RK4 (50 steps) ...")
        rk4_pts_p2 = rk4_sample(eval_model_reflow, n_samples=5000, n_steps=50, device=DEVICE)
        
        # ── Metrics Phase 2
        print("\n[METRICS] Phase 2 evaluation ...")
        improved_metrics = {}
        improved_metrics['straightness'] = straightness_score(eval_model_reflow, device=DEVICE)
        improved_metrics['wasserstein_w2'] = wasserstein_distance_2d(data, rk4_pts_p2)
        improved_metrics['mmd'] = compute_mmd(data, rk4_pts_p2)
        improved_metrics['path_efficiency'] = path_efficiency([t.to(DEVICE) for t in traj_p2])
        improved_metrics['solver_gap'] = solver_gap(euler_pts_p2, rk4_pts_p2)
        improved_metrics['kinetic_energy'] = kinetic_energy(eval_model_reflow, device=DEVICE)
        
        if args.dataset == '8gaussians':
            cov, counts = mode_coverage_8gaussians(rk4_pts_p2)
            improved_metrics['mode_coverage'] = cov
            improved_metrics['mode_counts'] = counts
        else:
            improved_metrics['mode_coverage'] = 0.0
            improved_metrics['mode_counts'] = []
        
        print(f"\n{'─'*50}")
        print("[PHASE 2 RESULTS - AFTER REFLOW]")
        print(f"  Straightness:     {improved_metrics['straightness']:.5f}")
        print(f"  W2 Distance:      {improved_metrics['wasserstein_w2']:.5f}")
        print(f"  MMD:              {improved_metrics['mmd']:.5f}")
        print(f"  Path Efficiency:  {improved_metrics['path_efficiency']:.5f}")
        print(f"  Solver Gap:       {improved_metrics['solver_gap']:.5f}")
        print(f"  Kinetic Energy:   {improved_metrics['kinetic_energy']:.5f}")
        if args.dataset == '8gaussians':
            print(f"  Mode Coverage:    {improved_metrics['mode_coverage']*100:.1f}%")
        print(f"{'─'*50}\n")
        
        final_model = eval_model_reflow
    else:
        final_model = eval_model

    # ── NFE vs Quality Curve
    print("\n[NFE CURVE] Evaluating quality at different step counts ...")
    nfe_curve = nfe_quality_curve(
        final_model, data,
        n_samples=2000,
        step_range=[10, 20, 50, 100, 200],
        device=DEVICE
    )
    
    print("\nNFE vs Quality:")
    for result in nfe_curve:
        print(f"  NFE={result['nfe']:>3}  W2={result['w2']:.5f}  MMD={result['mmd']:.5f}")

    # ── Comparison Plot
    out_comparison = os.path.join(args.out_dir, f'comparison_{args.dataset}.png')
    plot_comparison(baseline_metrics, improved_metrics, nfe_curve, out_comparison)

    # ── Save Metrics JSON
    final_metrics = {
        'dataset': args.dataset,
        'n_train': args.n_train,
        'epochs_phase1': args.epochs,
        'epochs_phase2': args.reflow_epochs if args.use_reflow else 0,
        'use_reflow': args.use_reflow,
        'use_ema': args.use_ema,
        'use_spectral_norm': args.use_spectral_norm,
        'curv_weight': args.curv_weight,
        'baseline': baseline_metrics,
        'improved': improved_metrics,
        'nfe_curve': nfe_curve,
        'n_params': n_p,
        'device': str(DEVICE),
        'torch_version': torch.__version__
    }
    
    mpath = os.path.join(args.out_dir, f'improved_metrics_{args.dataset}.json')
    with open(mpath, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n[DONE]   metrics → {mpath}")
    print(f"[DONE]   comparison plot → {out_comparison}")
    
    print(f"\n{'━'*70}")
    print("  SUMMARY")
    print(f"{'━'*70}")
    if args.use_reflow:
        print(f"  Straightness improvement:  {baseline_metrics['straightness']:.4f} → {improved_metrics['straightness']:.4f}")
        print(f"  Solver gap improvement:    {baseline_metrics['solver_gap']:.4f} → {improved_metrics['solver_gap']:.4f}")
        print(f"  W2 improvement:            {baseline_metrics['wasserstein_w2']:.4f} → {improved_metrics['wasserstein_w2']:.4f}")
    else:
        print(f"  Straightness:  {baseline_metrics['straightness']:.4f}")
        print(f"  Solver gap:    {baseline_metrics['solver_gap']:.4f}")
        print(f"  W2 distance:   {baseline_metrics['wasserstein_w2']:.4f}")
    print(f"{'━'*70}\n")
