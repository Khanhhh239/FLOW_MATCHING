"""
Flow Matching Phase 1: Prototype & Sanity Check
================================================
Conditional Flow Matching (CFM) — Lipman et al. 2022 / Albergo et al. 2023

Datasets  : 8-Gaussian mixture, 2-D Swiss Roll
Model     : Skip-MLP with sinusoidal time embedding
Samplers  : Euler (100 steps), RK4 (50 steps)
Metrics   : CFM loss, trajectory straightness, Wasserstein-2, MMD, 
            mode coverage, path efficiency, solver gap
Hardware  : RTX GPU (CUDA + TF32 + bfloat16 AMP) or CPU fallback

Fix v2    : Windows-safe torch.compile (disabled on CPU; guarded on GPU)
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

# bfloat16 AMP — stable on Ampere (sm_80+), safe on all
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
    angles  = torch.linspace(0, 2 * np.pi, 9)[:-1]          # 8 angles
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
# 2.  MODEL — VectorFieldMLP  u_θ(x, t)
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalEmbed(nn.Module):
    """
    Sinusoidal time embedding identical to DDPM / Ho et al. 2020.
    Maps scalar t ∈ [0,1] → ℝ^{dim} via sin/cos at log-spaced frequencies.
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        half  = dim // 2
        freqs = torch.exp(-np.log(10_000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)   # (half,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → (B, dim)
        x = t[:, None] * self.freqs[None, :] * 10
        return torch.cat([x.sin(), x.cos()], dim=-1)


class ResBlock(nn.Module):
    """Single residual block: Linear → LayerNorm → SiLU with skip."""
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
    u_θ : ℝ^D × [0,1] → ℝ^D

    Architecture
    ------------
    Input projection : [x ‖ embed(t)]  →  hidden
    Body             : depth × ResBlock(hidden)
    Skip connection  : input → hidden  (added before output)
    Output head      : hidden → D   (zero-init — starts at zero field)
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

        # Zero-init: model starts predicting u≡0, making early gradient clean
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)                    # (B, E)
        h  = torch.cat([x, te], dim=-1)            # (B, D+E)
        return self.output_proj(self.blocks(self.input_proj(h)) + self.skip(h))

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONDITIONAL FLOW MATCHING LOSS
# ─────────────────────────────────────────────────────────────────────────────
def cfm_loss(model: nn.Module,
             x1: torch.Tensor,
             sigma_min: float = 1e-4) -> torch.Tensor:
    """
    Conditional Flow Matching — optimal-transport (OT-CFM) formulation.

    Forward process (interpolant):
        x_t = (1 − (1−σ_min)·t)·x₀  +  t·x₁           … eq. (1)

    Marginal vector field (analytical target — constant along each path):
        u*(x_t | x₀, x₁) = x₁ − (1−σ_min)·x₀           … eq. (2)

    Training loss (regression on the vector field):
        L(θ) = 𝔼_{t∼U[0,1], x₀∼N(0,I), x₁∼p_data}
               [ ‖u_θ(x_t, t) − u*‖² ]                  … eq. (3)

    Because u* is constant along each path, perfectly trained trajectories
    are exactly straight lines — the key sanity-check invariant.
    """
    B      = x1.shape[0]
    device = x1.device
    dtype  = x1.dtype

    x0        = torch.randn_like(x1)                          # N(0, I)
    t         = torch.rand(B, device=device, dtype=dtype)     # U[0,1]
    t_exp     = t[:, None]                                    # (B,1) broadcast

    # Interpolate along the straight-line path
    x_t       = (1.0 - (1.0 - sigma_min) * t_exp) * x0 + t_exp * x1

    # Constant target vector field
    u_target  = x1 - (1.0 - sigma_min) * x0

    pred = model(x_t, t)

    loss = F.mse_loss(pred, u_target)

    mag_reg = 1e-4 * pred.norm(dim=-1).mean()

    return loss + mag_reg

# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model: nn.Module,
                data:  torch.Tensor,
                epochs:     int   = 500,
                batch_size: int   = 8192,
                lr:         float = 3e-4,
                device:     torch.device = DEVICE) -> list:

    model = model.to(device)

    # keep dataset on CPU
    data = data.float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP and AMP_DTYPE == torch.float16)
    warmup_epochs = 20

    # ── torch.compile: safe guard for Windows CPU (needs cl.exe) ──────────────
    use_compile = False

    # Windows + Triton often unstable → optional compile
    ENABLE_COMPILE = False

    if HAS_CUDA and ENABLE_COMPILE:
        try:
            torch._dynamo.config.suppress_errors = True

            model = torch.compile(model, mode="reduce-overhead")
            use_compile = True
            print("[torch.compile] Enabled")
        except Exception as exc:
            print(f"[torch.compile] Disabled ({exc})")
    else:
        print("[torch.compile] Disabled")

    loader = DataLoader(
        TensorDataset(data),
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
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    loss = cfm_loss(model, xb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = cfm_loss(model, xb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            ep_loss += loss.item()
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
            print(f"  Epoch {ep:>4}/{epochs}  loss={avg:.5f}  lr={lr_now:.2e}"
                  f"  t={time.time()-t0:.1f}s")

    return losses

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ODE SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def euler_sample(model, n_samples=5000, n_steps=100,
                 device=DEVICE, return_traj=False):
    """Explicit Euler ODE solver:  x_{t+dt} = x_t + dt · u_θ(x_t, t)"""
    model.eval()
    x   = torch.randn(n_samples, 2, device=device)
    dt  = 1.0 / n_steps
    # Collect ~21 snapshots evenly spaced along the trajectory
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
    """
    Classical 4th-order Runge-Kutta:
        k₁ = u(x_t,       t)
        k₂ = u(x_t+½dt·k₁, t+½dt)
        k₃ = u(x_t+½dt·k₂, t+½dt)
        k₄ = u(x_t+dt·k₃,  t+dt)
        x_{t+dt} = x_t + (dt/6)(k₁ + 2k₂ + 2k₃ + k₄)
    4× more accurate than Euler at the same number of function evaluations
    (halved steps), i.e. same NFE cost with O(dt⁴) vs O(dt) local error.
    """
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
# 6.  STRAIGHTNESS METRIC
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def straightness_score(model, n_samples=500, n_steps=200, device=DEVICE) -> float:
    """
    Quantifies how straight ODE trajectories are.

    For each particle path x₀ → x_T:
      chord  = ‖x_T − x₀‖
      deviation at step t = perpendicular distance from the chord line

    score = 1 − mean(mean_deviation / chord_length)

    Perfect score = 1.0  (all trajectories are exact straight lines).
    Typical well-trained CFM on 2-D toy data: ≥ 0.97.
    """
    model.eval()
    x0   = torch.randn(n_samples, 2, device=device)
    x    = x0.clone()
    dt   = 1.0 / n_steps
    traj = [x.clone()]

    for i in range(n_steps):
        t  = torch.full((n_samples,), i / n_steps, device=device)
        x  = x + model(x, t) * dt
        traj.append(x.clone())

    traj  = torch.stack(traj, dim=0)          # (T+1, N, 2)
    start = traj[0]                            # (N, 2)
    end   = traj[-1]
    chord = (end - start).norm(dim=-1).clamp(min=1e-6)     # (N,)

    devs = []
    for step in traj[1:-1]:
        delta     = step - start                            # (N, 2)
        chord_dir = (end - start) / chord[:, None]         # unit vector
        proj      = (delta * chord_dir).sum(-1, keepdim=True) * chord_dir
        perp      = (delta - proj).norm(dim=-1)            # (N,)
        devs.append(perp)

    mean_dev = torch.stack(devs).mean()
    return float(1.0 - (mean_dev / chord.mean()).item())

@torch.no_grad()
def wasserstein_distance_2d(real: torch.Tensor,
                            fake: torch.Tensor,
                            max_samples: int = 1000) -> float:
    """
    Approximate Wasserstein-2 distance using Hungarian matching.
    
    Lower = better.
    Very meaningful for Flow Matching / OT models.
    """
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
    """
    Maximum Mean Discrepancy with RBF kernel.
    
    Lower = better.
    """
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
def mode_coverage_8gaussians(samples: torch.Tensor,
                             radius: float = 0.5):
    """
    Measures how many of the 8 Gaussian modes are covered.
    
    Returns:
        coverage_ratio,
        counts_per_mode
    """
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
    """
    Ratio:
        straight-line distance / actual trajectory length
    
    Perfect straight path = 1.0
    """
    traj = torch.stack(traj, dim=0)  # (T,N,2)
    
    diffs = traj[1:] - traj[:-1]
    path_len = diffs.norm(dim=-1).sum(0)
    
    chord = (traj[-1] - traj[0]).norm(dim=-1)
    
    eff = (chord / (path_len + 1e-8)).mean()
    
    return float(eff.item())


@torch.no_grad()
def solver_gap(euler_samples, rk4_samples):
    """
    Difference between Euler and RK4 generated distributions.
    
    Lower = vector field easier to integrate.
    """
    n = min(len(euler_samples), len(rk4_samples))
    
    gap = ((euler_samples[:n] - rk4_samples[:n])**2).sum(dim=-1).sqrt().mean()
    
    return float(gap.item())


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(target_data, euler_samples, rk4_samples, traj, losses,
             dataset_label, score, out_path, metrics_dict=None):

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 20), facecolor='#0a0a0c')
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.35, wspace=0.28,
                            left=0.05, right=0.97, top=0.94, bottom=0.04)

    C_BLUE  = '#00e5ff'
    C_PINK  = '#ff4081'
    C_GREEN = '#76ff03'
    C_YELLOW = '#ffeb3b'
    C_PANEL = '#13131a'
    CMTGT   = LinearSegmentedColormap.from_list('t', ['#001433','#0055cc','#00e5ff'])
    CMSMP   = LinearSegmentedColormap.from_list('s', ['#1a001a','#cc0066','#ff4081'])

    def style_ax(ax):
        ax.set_facecolor(C_PANEL)
        for sp in ax.spines.values(): sp.set_color('#252530')
        ax.tick_params(colors='#667788', labelsize=8)
        ax.xaxis.label.set_color('#8899aa'); ax.yaxis.label.set_color('#8899aa')

    def density_plot(ax, pts, cmap, dot_color, title):
        style_ax(ax)
        try:
            k   = gaussian_kde(pts[:5000].T)
            lo  = pts.min(0) - 0.3;  hi = pts.max(0) + 0.3
            xi  = np.linspace(lo[0], hi[0], 200)
            yi  = np.linspace(lo[1], hi[1], 200)
            Xi, Yi = np.meshgrid(xi, yi)
            ax.contourf(Xi, Yi, k(np.vstack([Xi.ravel(),Yi.ravel()])).reshape(Xi.shape),
                        levels=30, cmap=cmap, alpha=0.85)
        except Exception: pass
        ax.scatter(pts[::10,0], pts[::10,1], s=1.5, c=dot_color, alpha=0.3)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')

    # 1 – target
    density_plot(fig.add_subplot(gs[0,0]),
                 target_data[:5000].numpy(), CMTGT, C_BLUE,
                 f'Target: {dataset_label}')

    # 2 – Euler
    density_plot(fig.add_subplot(gs[0,1]),
                 euler_samples[:5000].numpy(), CMSMP, C_PINK,
                 'Euler samples (100 steps)')

    # 3 – RK4
    density_plot(fig.add_subplot(gs[0,2]),
                 rk4_samples[:5000].numpy(), CMSMP, C_GREEN,
                 'RK4 samples (50 steps)')

    # 4 – loss curve
    ax4 = fig.add_subplot(gs[0,3]);  style_ax(ax4)
    ep  = np.arange(1, len(losses)+1)
    ax4.fill_between(ep, losses, alpha=0.18, color=C_BLUE)
    ax4.plot(ep, losses, color=C_BLUE, lw=1.4, label='per-epoch')
    if len(losses) >= 20:
        w = max(1, len(losses)//40)
        sm = np.convolve(losses, np.ones(w)/w, mode='valid')
        ax4.plot(np.arange(w, len(losses)+1), sm, color=C_PINK, lw=2.2, label='smoothed')
    ax4.set_yscale('log'); ax4.set_title('Training loss (MSE)', color='white',
                                          fontsize=10, fontweight='bold')
    ax4.set_xlabel('Epoch'); ax4.legend(fontsize=8, facecolor=C_PANEL, labelcolor='white')

    # 5 – trajectories
    ax5  = fig.add_subplot(gs[1,0:2]);  style_ax(ax5)
    traj_np = [t.numpy() for t in traj]
    N_SHOW  = 24
    palette = plt.cm.plasma(np.linspace(0, 1, N_SHOW))
    for i in range(N_SHOW):
        pts = np.array([step[i] for step in traj_np])
        ax5.plot(pts[:,0], pts[:,1], color=palette[i], lw=1.1, alpha=0.85)
        ax5.scatter(pts[0,0],  pts[0,1],  s=18, color='white',  zorder=5)
        ax5.scatter(pts[-1,0], pts[-1,1], s=22, color=C_PINK,   zorder=5, marker='*')
        s, e = traj_np[0][i], traj_np[-1][i]
        ax5.plot([s[0],e[0]], [s[1],e[1]], color='white', lw=0.6, alpha=0.22, ls='--')
    ax5.set_title(f'ODE trajectories  |  Straightness = {score:.4f}  '
                  f'(1.0 = perfect)', color='white', fontsize=10, fontweight='bold')
    ax5.set_aspect('equal')

    # 6 – vector field quiver at t=0.5
    ax6 = fig.add_subplot(gs[1,2]);  style_ax(ax6)
    lim = 3.5
    gx  = np.linspace(-lim, lim, 18)
    Xi, Yi = np.meshgrid(gx, gx)
    gpts = torch.tensor(np.stack([Xi.ravel(), Yi.ravel()], 1),
                        dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        vf = model(gpts,
                   torch.full((gpts.shape[0],), .5, device=DEVICE)).cpu().numpy()
    Vx, Vy = vf[:,0].reshape(Xi.shape), vf[:,1].reshape(Yi.shape)
    mag = np.hypot(Vx, Vy) + 1e-8
    ax6.quiver(Xi, Yi, Vx/mag, Vy/mag, mag,
               cmap='plasma', scale=26, width=0.004, alpha=0.85)
    ax6.set_title('Vector field  u_θ(x, t=0.5)', color='white',
                  fontsize=10, fontweight='bold')
    ax6.set_xlim(-lim,lim); ax6.set_ylim(-lim,lim); ax6.set_aspect('equal')

    # 7 – density evolution
    ax7 = fig.add_subplot(gs[1,3]);  style_ax(ax7)
    T   = len(traj_np)
    for k, idx in enumerate([0, T//4, T//2, 3*T//4, T-1]):
        pts = traj_np[idx]
        c   = plt.cm.cool(k / 4)
        ax7.scatter(pts[:,0], pts[:,1], s=5, color=c, alpha=0.55,
                    label=f't={k/4:.2f}')
    ax7.set_title('Density evolution  t: 0 → 1', color='white',
                  fontsize=10, fontweight='bold')
    ax7.legend(fontsize=7, facecolor=C_PANEL, labelcolor='white', markerscale=1.5)
    ax7.set_aspect('equal')

    # ── ROW 3: METRICS VISUALIZATIONS ──
    if metrics_dict:
        # 8 – Mode coverage histogram (8-gaussians only)
        ax8 = fig.add_subplot(gs[2,0]);  style_ax(ax8)
        if 'mode_counts' in metrics_dict and metrics_dict['mode_counts']:
            counts = metrics_dict['mode_counts']
            bars = ax8.bar(np.arange(8), counts, color=C_BLUE, alpha=0.8, edgecolor='white', linewidth=0.5)
            ax8.axhline(20, color=C_PINK, ls='--', lw=1, alpha=0.6, label='Detection threshold')
            ax8.set_xlabel('Mode index', fontsize=9)
            ax8.set_ylabel('Sample count', fontsize=9)
            ax8.set_title(f'Mode Coverage: {metrics_dict.get("mode_coverage", 0)*100:.1f}%', 
                         color='white', fontsize=10, fontweight='bold')
            ax8.legend(fontsize=7, facecolor=C_PANEL, labelcolor='white')
            ax8.set_xticks(range(8))
        else:
            ax8.text(0.5, 0.5, 'Mode coverage\n(8-gaussians only)', 
                    ha='center', va='center', color='#667788', fontsize=9,
                    transform=ax8.transAxes)
            ax8.set_title('Mode Coverage', color='white', fontsize=10, fontweight='bold')

        # 9 – Euler vs RK4 overlay
        ax9 = fig.add_subplot(gs[2,1]);  style_ax(ax9)
        n_show = 1000
        ax9.scatter(euler_samples[:n_show,0], euler_samples[:n_show,1], 
                   s=3, c=C_BLUE, alpha=0.4, label='Euler')
        ax9.scatter(rk4_samples[:n_show,0], rk4_samples[:n_show,1], 
                   s=3, c=C_PINK, alpha=0.4, label='RK4')
        ax9.set_title(f'Solver Comparison (gap={metrics_dict.get("solver_gap", 0):.4f})', 
                     color='white', fontsize=10, fontweight='bold')
        ax9.legend(fontsize=8, facecolor=C_PANEL, labelcolor='white')
        ax9.set_aspect('equal')

        # 10 – Path efficiency histogram
        ax10 = fig.add_subplot(gs[2,2]);  style_ax(ax10)
        if traj:
            traj_tensor = torch.stack(traj, dim=0)  # (T,N,2)
            diffs = traj_tensor[1:] - traj_tensor[:-1]
            path_lens = diffs.norm(dim=-1).sum(0)
            chords = (traj_tensor[-1] - traj_tensor[0]).norm(dim=-1)
            efficiencies = (chords / (path_lens + 1e-8)).numpy()
            
            ax10.hist(efficiencies, bins=30, color=C_GREEN, alpha=0.7, edgecolor='white', linewidth=0.5)
            ax10.axvline(efficiencies.mean(), color=C_PINK, ls='--', lw=2, 
                        label=f'Mean={efficiencies.mean():.3f}')
            ax10.axvline(1.0, color=C_YELLOW, ls=':', lw=2, alpha=0.6, label='Perfect=1.0')
            ax10.set_xlabel('Chord / Path length', fontsize=9)
            ax10.set_ylabel('Count', fontsize=9)
            ax10.set_title(f'Path Efficiency Distribution', 
                          color='white', fontsize=10, fontweight='bold')
            ax10.legend(fontsize=7, facecolor=C_PANEL, labelcolor='white')

        # 11 – Metrics summary panel
        ax11 = fig.add_subplot(gs[2,3]);  style_ax(ax11)
        ax11.axis('off')
        
        metrics_text = "PAPER-LEVEL METRICS\n" + "─"*30 + "\n\n"
        
        if 'wasserstein_w2' in metrics_dict:
            w2_val = metrics_dict['wasserstein_w2']
            w2_color = C_GREEN if w2_val < 0.1 else (C_YELLOW if w2_val < 0.3 else C_PINK)
            metrics_text += f"Wasserstein-2:  {w2_val:.5f}\n"
        
        if 'mmd' in metrics_dict:
            mmd_val = metrics_dict['mmd']
            mmd_color = C_GREEN if mmd_val < 0.01 else (C_YELLOW if mmd_val < 0.05 else C_PINK)
            metrics_text += f"MMD:            {mmd_val:.5f}\n"
        
        if 'mode_coverage' in metrics_dict:
            cov_val = metrics_dict['mode_coverage']
            cov_color = C_GREEN if cov_val > 0.9 else (C_YELLOW if cov_val > 0.7 else C_PINK)
            metrics_text += f"Mode Coverage:  {cov_val*100:.1f}%\n"
        
        if 'path_efficiency' in metrics_dict:
            eff_val = metrics_dict['path_efficiency']
            eff_color = C_GREEN if eff_val > 0.95 else (C_YELLOW if eff_val > 0.90 else C_PINK)
            metrics_text += f"Path Efficiency: {eff_val:.5f}\n"
        
        if 'solver_gap' in metrics_dict:
            gap_val = metrics_dict['solver_gap']
            gap_color = C_GREEN if gap_val < 0.05 else (C_YELLOW if gap_val < 0.15 else C_PINK)
            metrics_text += f"Solver Gap:     {gap_val:.5f}\n"
        
        metrics_text += f"\nStraightness:   {score:.5f}\n"
        metrics_text += "\n" + "─"*30 + "\n"
        metrics_text += "Lower is better: W2, MMD, Gap\n"
        metrics_text += "Higher is better: Coverage, Efficiency"
        
        ax11.text(0.1, 0.95, metrics_text, 
                 transform=ax11.transAxes,
                 fontsize=10, 
                 verticalalignment='top',
                 fontfamily='monospace',
                 color='white',
                 bbox=dict(boxstyle='round', facecolor=C_PANEL, alpha=0.8, edgecolor='#252530'))

    fig.suptitle(
        f'Flow Matching — Phase 1 Sanity Check  ·  {dataset_label}'
        f'  ·  Straightness = {score:.4f}',
        color='white', fontsize=13, fontweight='bold', y=0.97)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0c')
    plt.close(fig)
    print(f"[PLOT]  saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset',   default='8gaussians',
                   choices=['8gaussians', 'swiss_roll'])
    p.add_argument('--n_train',   type=int,   default=50_000)
    p.add_argument('--epochs',    type=int,   default=500)
    p.add_argument('--batch',     type=int,   default=8192)
    p.add_argument('--hidden',    type=int,   default=256)
    p.add_argument('--depth',     type=int,   default=5)
    p.add_argument('--lr',        type=float, default=3e-4)
    p.add_argument('--sigma_min', type=float, default=0.05,
                   help='Minimum noise floor σ_min (keeps paths from collapsing)')
    p.add_argument('--out_dir',   default='outputs')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'━'*58}")
    print(f"  Flow Matching Phase 1  ·  dataset={args.dataset}")
    print(f"  Python {sys.version.split()[0]}  ·  "
          f"PyTorch {torch.__version__}  ·  {platform.system()}")
    print(f"{'━'*58}")

    # ── Data
    if args.dataset == '8gaussians':
        data, label = make_8gaussians(args.n_train),  '8-Gaussian Mixture'
    else:
        data, label = make_swiss_roll_2d(args.n_train), 'Swiss Roll 2-D'
    print(f"[DATA]   {data.shape[0]:,} samples  "
          f"μ={data.mean(0).numpy().round(4)}  σ={data.std(0).numpy().round(4)}")

    # ── Model
    model = VectorFieldMLP(hidden=args.hidden, depth=args.depth)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"[MODEL]  {n_p:,} params  hidden={args.hidden}  depth={args.depth}"
          f"  AMP={USE_AMP}")

    # ── Train
    print(f"\n[TRAIN]  {args.epochs} epochs  batch={args.batch}  lr={args.lr}")
    losses = train_model(model, data,
                         epochs=args.epochs, batch_size=args.batch,
                         lr=args.lr, device=DEVICE)

    # ── Sample
    print("\n[SAMPLE] Euler (100 steps) ...")
    euler_pts, traj = euler_sample(model, n_samples=5000, n_steps=100,
                                   device=DEVICE, return_traj=True)
    print("[SAMPLE] RK4   (50 steps)  ...")
    rk4_pts = rk4_sample(model, n_samples=5000, n_steps=50, device=DEVICE)

    # ── Straightness
    print("[METRIC] Straightness score ...")
    score = straightness_score(model, n_samples=500, n_steps=200, device=DEVICE)
    print(f"[METRIC] Straightness = {score:.5f}  (perfect = 1.0)")

    # ── Additional Metrics
    print("[METRIC] Wasserstein distance ...")
    w2 = wasserstein_distance_2d(data, rk4_pts)
    
    print("[METRIC] MMD ...")
    mmd = compute_mmd(data, rk4_pts)
    
    print("[METRIC] Mode coverage ...")
    if args.dataset == '8gaussians':
        coverage, counts = mode_coverage_8gaussians(rk4_pts)
    else:
        coverage, counts = 0.0, []
    
    print("[METRIC] Path efficiency ...")
    efficiency = path_efficiency([t.to(DEVICE) for t in traj])
    
    print("[METRIC] Euler/RK4 solver gap ...")
    gap = solver_gap(euler_pts, rk4_pts)
    
    print(f"\n{'─'*50}")
    print(f"[METRIC] W2 distance      = {w2:.5f}")
    print(f"[METRIC] MMD              = {mmd:.5f}")
    print(f"[METRIC] Mode coverage    = {coverage*100:.1f}%")
    print(f"[METRIC] Path efficiency  = {efficiency:.5f}")
    print(f"[METRIC] Solver gap       = {gap:.5f}")
    if counts:
        print(f"[METRIC] Per-mode counts  = {counts}")
    print(f"{'─'*50}\n")

    # ── Plot
    metrics_for_plot = {
        'wasserstein_w2': w2,
        'mmd': mmd,
        'mode_coverage': coverage,
        'mode_counts': counts,
        'path_efficiency': efficiency,
        'solver_gap': gap
    }
    
    out_img = os.path.join(args.out_dir, f'flow_matching_{args.dataset}.png')
    plot_all(data[:5000], euler_pts, rk4_pts, traj,
             losses, label, score, out_img, metrics_for_plot)

    # ── Metrics JSON
    metrics = dict(dataset=args.dataset, n_train=args.n_train,
                   hidden=args.hidden, depth=args.depth,
                   epochs=args.epochs, batch=args.batch,
                   final_loss=round(float(losses[-1]), 6),
                   best_loss=round(float(min(losses)), 6),
                   straightness=round(score, 5),
                   wasserstein_w2=round(w2, 6),
                   mmd=round(mmd, 6),
                   mode_coverage=round(coverage, 4),
                   path_efficiency=round(efficiency, 5),
                   solver_gap=round(gap, 5),
                   n_params=n_p,
                   device=str(DEVICE), amp=USE_AMP,
                   torch_version=torch.__version__)
    mpath = os.path.join(args.out_dir, f'metrics_{args.dataset}.json')
    with open(mpath, 'w') as f: json.dump(metrics, f, indent=2)
    print(f"\n[DONE]   metrics → {mpath}")
    print(json.dumps(metrics, indent=2))
