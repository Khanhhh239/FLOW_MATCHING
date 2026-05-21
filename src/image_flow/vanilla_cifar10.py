"""
flow_matching_phase2_cifar10.py
================================
Phase 2 (run in LOCAL) — use dataset CIFAR-10 (download through torchvision).

Don't need:
  - Kaggle
  - diffusers / VAE

  Need set up:
  pip install torch torchvision matplotlib tqdm einops

Pipeline:
  Step 1 — Pre-compute "latents" (in here = normalize pixel space, don't utilize VAE because images in this dataset has 32x32 pixel (low-dim)).
        With high-dim images such as 512x512, 1024x1024, we need use VAE to reduce into low-dim
           Save into disk with the format .pt to train quickly
  Step 2 — Train Mini-DiT with CFM loss (class-conditional, CFG dropout)
  Step 3 — Inference: Euler / RK4 at 10 / 20 / 50 steps → compare NFE

  Run:
  # Entire pipeline
  python flow_matching_phase2_cifar10.py

  # Only train
  python flow_matching_phase2_cifar10.py --mode train --epochs 100

  # Only inference (need train)
  python flow_matching_phase2_cifar10.py --mode infer --checkpoint outputs/best.pt

Hardware:
  RTX GPU  → ~25 min / 100 epochs, batch=256
  CPU only → ~3 min / 5 epochs (use --epochs 5 --batch 128 to quick test)
"""

import os, sys, json, time, argparse, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure Vietnamese logs print safely on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# 0. DEVICE
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark     = True
        torch.backends.cuda.matmul.allow_tf32 = True
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU]    {name}  |  {vram:.1f} GB VRAM")
        return torch.device('cuda')
    print("[DEVICE] CPU — chạy được nhưng chậm hơn GPU ~10x")
    return torch.device('cpu')

DEVICE = get_device()

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

# ─────────────────────────────────────────────────────────────────────────────
# 1. STEP 1 — PRECOMPUTE "LATENTS" From CIFAR-10
#    Don't use VAE — normalize img to [-1,1], save tensor .pt
#    Since CIFAR-10 small (32×32) should use pixel space directly
# ─────────────────────────────────────────────────────────────────────────────
def step1_precompute(out_dir: str = 'outputs/latents',
                     data_dir: str = './data') -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / 'metadata.json'
    if meta_path.exists():
        print(f"[STEP1] Latents is available in {out_dir}/ — skip precompute")
        try:
            with open(meta_path, encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[WARN] metadata.json error format, precompute...")
            meta_path.unlink(missing_ok=True)

    print(f"\n{'─'*50}")
    print(f"  STEP 1: Precompute CIFAR-10 latents")
    print(f"{'─'*50}")

    # Download CIFAR-10 automatically (fallback synthetic if connection is interrupt)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    try:
        train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf)
        val_ds   = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
        print(f"[DATA]  CIFAR-10  train={len(train_ds):,}  val={len(val_ds):,}")

        def save_split(ds, split_name):
            imgs_list, labels_list = [], []
            loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
            for imgs, labels in tqdm(loader, desc=f"  Cache {split_name}"):
                imgs_list.append(imgs); labels_list.append(labels)
            imgs_all   = torch.cat(imgs_list)
            labels_all = torch.cat(labels_list)
            torch.save(imgs_all,   out_dir / f'{split_name}_imgs.pt')
            torch.save(labels_all, out_dir / f'{split_name}_labels.pt')
            print(f"  {split_name}: {imgs_all.shape}  →  {out_dir}/{split_name}_*.pt")
            return len(ds)

        n_train = save_split(train_ds, 'train')
        n_val   = save_split(val_ds,   'val')

    except Exception as e:
        print(f"[WARN]  Failed to download CIFAR-10: {e}")
        print(f"[INFO]  Create synthetic data (3×32×32, 10 classes) to test pipeline...")
        # Synthetic: 5000 train, 1000 val
        torch.manual_seed(42)
        for split, n in [("train", 5000), ("val", 1000)]:
            imgs   = torch.randn(n, 3, 32, 32).clamp(-1, 1)
            labels = torch.randint(0, 10, (n,))
            torch.save(imgs,   out_dir / f'{split}_imgs.pt')
            torch.save(labels, out_dir / f'{split}_labels.pt')
            print(f"  {split}: synthetic {imgs.shape}")
        n_train, n_val = 5000, 1000

    meta = {
        'dataset':      'CIFAR-10',
        'n_train':      n_train,
        'n_val':        n_val,
        'img_shape':    [3, 32, 32],  # C, H, W — pixel space (not VAE)
        'latent_shape': [3, 32, 32],  # same img_shape because of not VAE
        'classes':      CIFAR10_CLASSES,
        'n_classes':    10,
        'pixel_range':  '[-1, 1]',
        'note':         'pixel space directly',
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[DONE]  metadata → {meta_path}")
    return meta


class CachedLatentDataset(Dataset):
    """Load pre-computed tensor from disk, faster than decode from images for each epoches."""
    def __init__(self, out_dir: str, split: str = 'train'):
        out_dir = Path(out_dir)
        self.imgs   = torch.load(out_dir / f'{split}_imgs.pt',   map_location='cpu')
        self.labels = torch.load(out_dir / f'{split}_labels.pt', map_location='cpu')

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MINI-DiT MODEL
# ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """t: (B,) → (B, dim)"""
    half  = dim // 2
    freqs = torch.exp(-math.log(10_000) *
                      torch.arange(half, device=t.device, dtype=t.dtype) / half)
    x = t[:, None] * freqs[None] * 1000
    return torch.cat([x.sin(), x.cos()], dim=-1)
    # result : embedding(t) = cat[sin(w_i.1000t), cos(w_i.1000t)] with i from 1 to half-1
    # With: angle frequency w_i = 1000^(-i/half)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class PatchEmbed(nn.Module):
    """Divide images into patch, project in token space."""
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        # Each patch isn't overlapping and the channel increase to 256

    def forward(self, x):
        # x: (B, C, H, W) → (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class DiTBlock(nn.Module):
    """Transformer block với adaLN-Zero conditioning."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        # Initialize the LayerNorm, turn off default scale and default bias because it will be learned automatically basing on condition (c) through modulate
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        # Attention block to help all patchs connect with together, dropout (that could prevent overfitting) but in generative task, remove some 
        # tight relations causing the chaos of Attention Weights -> Images has harmful issues like noisy, not fine-grained, not continous
         
        self.attn  = nn.MultiheadAttention(dim, n_heads,
                                            batch_first=True, dropout=0.0)
        # Standard MLP : broad to 4*dim, feed to GeLU, scale down dim -> handle and enrich several features of each independent image tokens (patchs)
        # GeLU = x.Gaussian(x)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        # adaLN-Zero: 6 params (shift/scale/gate × 2 layers), SiLU = x.sigmoid(x)
        # Intialize scale and bias by 0 in the last layer -> identity function : f(x)=x straightforward signal helping stable training
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1) #chunk into 6 sequences
        # Self-attention
        h, _ = self.attn(modulate(self.norm1(x), s1, sc1),
                         modulate(self.norm1(x), s1, sc1),
                         modulate(self.norm1(x), s1, sc1), need_weights=False)
        x = x + g1[:, None] * h
        # MLP
        x = x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class MiniDiT(nn.Module):
    """
    Mini DiT cho CIFAR-10 (32×32×3).
    Patch size = 4 → 8×8 = 64 tokens each images.

    Configs:
      tiny : depth=4,  dim=128  → 3.5M params  (CPU test)
      small: depth=6,  dim=256  → 14M  params  (RTX laptop)
      base : depth=8,  dim=384  → 33M  params  (RTX 3080+)
    """
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 dim=256, depth=6, n_heads=8, n_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch      = in_ch
        self.n_patches  = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, dim) # patchs -> vector (dim = 256)
        self.pos_embed   = nn.Parameter(torch.randn(1, self.n_patches, dim) * 0.02) #trainable matrix to encode possition

        # Time + class conditioning
        # MLP convert embedding time into corresponding vector (with images)
        self.time_mlp  = nn.Sequential(
            nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim)
        ) 
        self.class_emb = nn.Embedding(n_classes + 1, dim)  # +1 = null (CFG : Classifer-Free Guidance)
        self.null_cls  = n_classes

        self.blocks = nn.ModuleList([
            DiTBlock(dim, n_heads) for _ in range(depth)
        ])

        # Final: adaLN + project patch → pixels
        self.final_norm   = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada    = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_linear = nn.Linear(dim, patch_size * patch_size * in_ch) # return the initial img size

        # Zero-init output → starts predicting zero field
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MiniDiT] dim={dim} depth={depth} heads={n_heads} "
              f"patches={self.n_patches} params={n_params/1e6:.1f}M")

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        if y is None:
            y = torch.full((B,), self.null_cls, device=x.device, dtype=torch.long) # unconditional generation

        # Tokenise
        tok = self.patch_embed(x) + self.pos_embed  # (B, N, D)

        # Conditioning
        te = sinusoidal_embed(t, dim=128)
        c  = self.time_mlp(te) + self.class_emb(y)  # (B, D)

        for block in self.blocks:
            tok = block(tok, c) # Iterate 6 transformer blocks 

        # Output
        s, sc = self.final_ada(c).chunk(2, dim=-1) #gain learnable shift and scale
        tok   = modulate(self.final_norm(tok), s, sc) # Implement adaptive Layer normalization
        tok   = self.final_linear(tok)  # (B, N, p²·C), project into token dim

        # Unpatchify → (B, C, H, W), Arrange patchs into result image
        p = self.patch_size
        h = w = int(self.n_patches ** 0.5)
        tok = tok.reshape(B, h, w, p, p, self.in_ch)
        tok = torch.einsum('bhwpqc->bchpwq', tok) # transform the order dimension
        return tok.reshape(B, self.in_ch, h*p, w*p)

    def forward_cfg(self, x, t, y, guidance=4.0):
        """Classifier-Free Guidance: combine cond + uncond."""
        # This function use for inference step
        B    = x.shape[0]
        null = torch.full_like(y, self.null_cls)
        # Instead of run twice, one for condition and one for uncondition, we concatenate both and run once to optimize speed (powerfull hardware)
        x2   = torch.cat([x, x])
        t2   = torch.cat([t, t])
        y2   = torch.cat([y, null])
        u    = self.forward(x2, t2, y2)
        u_c, u_u = u.chunk(2)
        return u_u + guidance * (u_c - u_u)


CONFIGS = {
    'tiny':  dict(dim=128, depth=4, n_heads=4),   # 3.5M — CPU ok
    'small': dict(dim=256, depth=6, n_heads=8),   # 14M  — RTX laptop
    'base':  dict(dim=384, depth=8, n_heads=8),   # 33M  — RTX 3080+
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CFM LOSS
# ─────────────────────────────────────────────────────────────────────────────
def cfm_loss(model, x1, y, sigma_min=1e-4, cfg_dropout=0.1):
    """
    x1 : real images
    cfg_dropout : ratio of unconditional image
    OT-CFM loss — same Phase 1, now in image 3×32×32.

    x_t    = (1 - (1-σ)t)·x0  +  t·x1
    u*     = x1 - (1-σ)·x0        ← don't depend on time
    L(θ)   = E‖u_θ(x_t,t,y) - u*‖²
    """
    B  = x1.shape[0]
    x0 = torch.randn_like(x1) # noisy image
    t  = torch.rand(B, device=x1.device, dtype=x1.dtype) # select t from uniform distribution
    t4 = t.view(B, 1, 1, 1) # use for broadcasting

    x_t    = (1 - (1 - sigma_min) * t4) * x0 + t4 * x1
    u_star = x1 - (1 - sigma_min) * x0

    # CFG dropout
    if cfg_dropout > 0:
        mask    = torch.rand(B, device=x1.device) < cfg_dropout
        y       = y.clone()
        y[mask] = model.null_cls

    return F.mse_loss(model(x_t, t, y), u_star)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EMA
# ─────────────────────────────────────────────────────────────────────────────
#Exponential Moving Average 
class EMA:
    def __init__(self, model, decay=0.9999): # Remain 0,999 result from prior weights
        self.decay  = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach() * (1 - self.decay))
            #Update weight : W_t_shadow = decay.W_(t-1)_shadow + (1 - decay).W_(t-1)_online
            # Weight main model:   ──/\──/\──/\──/\──  (high slope, fluctuate, noise)
            # Weight EMA (Shadow):   ──────────────────  (Smooth line)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


# ─────────────────────────────────────────────────────────────────────────────
# 5. STEP 2 — TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def step2_train(latent_dir, out_dir, config='small', epochs=100,
                batch=256, lr=1e-4, cfg_dropout=0.1,
                val_every=10, save_every=50):

    out_dir  = Path(out_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'; ckpt_dir.mkdir(exist_ok=True)

    print(f"\n{'─'*50}")
    print(f"  STEP 2: Train Mini-DiT in CIFAR-10")
    print(f"{'─'*50}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = CachedLatentDataset(latent_dir, 'train')
    val_ds   = CachedLatentDataset(latent_dir, 'val')
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE.type=='cuda'))
    val_ld   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE.type=='cuda'))
    print(f"[DATA]  train={len(train_ds):,}  val={len(val_ds):,}  batch={batch}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MiniDiT(n_classes=10, **CONFIGS[config]).to(DEVICE)

    # torch.compile: skip on Windows due to frequent Triton issues.
    if DEVICE.type == 'cuda' and os.name != 'nt':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("[compile] Enabled")
        except Exception as e:
            print(f"[compile] Skipped: {e}")
    elif DEVICE.type == 'cuda':
        print("[compile] Skipped on Windows (avoid Triton runtime errors)")

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=1e-4, betas=(0.9, 0.99))
    total = epochs * len(train_ld)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total,
        pct_start=0.05, anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    ema    = EMA(model, decay=0.9999)

    # ── Train loop ────────────────────────────────────────────────────────────
    log, best_val = [], float('inf')
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0

        for x1, y in tqdm(train_ld, desc=f"Ep {ep}/{epochs}", leave=False):
            x1 = x1.to(DEVICE);  y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type,
                                 dtype=torch.bfloat16,
                                 enabled=(DEVICE.type == 'cuda')):
                loss = cfm_loss(model, x1, y, cfg_dropout=cfg_dropout)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt);  scaler.update()
            sched.step();  ema.update(model)
            ep_loss += loss.item()

        avg_train = ep_loss / len(train_ld)

        # ── Validation ────────────────────────────────────────────────────────
        if ep % val_every == 0 or ep == epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x1, y in val_ld:
                    x1, y = x1.to(DEVICE), y.to(DEVICE)
                    with torch.autocast(device_type=DEVICE.type,
                                         dtype=torch.bfloat16,
                                         enabled=(DEVICE.type=='cuda')):
                        val_loss += cfm_loss(model, x1, y, cfg_dropout=0).item()
            avg_val = val_loss / len(val_ld)
            elapsed = time.time() - t0
            print(f"  Ep {ep:>4}  train={avg_train:.4f}  val={avg_val:.4f}"
                  f"  lr={sched.get_last_lr()[0]:.2e}  {elapsed:.0f}s")
            log.append({'epoch': ep, 'train': avg_train, 'val': avg_val})

            if avg_val < best_val:
                best_val = avg_val
                torch.save({
                    'epoch': ep, 'model': model.state_dict(),
                    'ema': ema.shadow, 'val_loss': avg_val,
                    'config': config, 'classes': CIFAR10_CLASSES,
                }, ckpt_dir / 'best.pt')
                print(f"  [CKPT] best.pt  val={avg_val:.4f}")

        if ep % save_every == 0:
            torch.save({'epoch': ep, 'model': model.state_dict(),
                        'ema': ema.shadow},
                       ckpt_dir / f'epoch_{ep:04d}.pt')

    with open(out_dir / 'train_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)
    print(f"\n[DONE]  best_val={best_val:.4f}  total={time.time()-t0:.0f}s")
    return ckpt_dir / 'best.pt'


# ─────────────────────────────────────────────────────────────────────────────
# 6. ODE SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def euler_sample(model, n, y, n_steps=50, guidance=4.0, device=DEVICE):
    """Euler: x_{t+dt} = x_t + dt·u_θ(x_t,t,y)  |  NFE = n_steps"""
    model.eval()
    x  = torch.randn(n, 3, 32, 32, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i / n_steps, device=device)
        x = x + dt * model.forward_cfg(x, t, y, guidance=guidance)
    return x


@torch.no_grad()
def rk4_sample(model, n, y, n_steps=20, guidance=4.0, device=DEVICE):
    """RK4: O(dt⁴) error  |  NFE = 4×n_steps"""
    model.eval()
    x  = torch.randn(n, 3, 32, 32, device=device)
    dt = 1.0 / n_steps

    def vf(xt, ti):
        t = torch.full((n,), ti, device=device)
        return model.forward_cfg(xt, t, y, guidance=guidance)

    for i in range(n_steps):
        ti = i / n_steps
        k1 = vf(x,               ti)
        k2 = vf(x + .5*dt*k1,   ti + .5*dt)
        k3 = vf(x + .5*dt*k2,   ti + .5*dt)
        k4 = vf(x + dt*k3,      ti + dt)
        x  = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x


def to_img(x: torch.Tensor) -> np.ndarray:
    """Tensor [-1,1] → uint8 numpy (H,W,3)"""
    x = x.float().clamp(-1, 1)
    x = (x + 1) / 2 * 255
    return x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 7. STEP 3 — INFERENCE + NFE COMPARISON GRID
# ─────────────────────────────────────────────────────────────────────────────
def step3_infer(checkpoint_path: str, out_dir: str,
                guidance: float = 4.0, n_per_class: int = 4):

    out_dir = Path(out_dir);  out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*50}")
    print(f"  STEP 3: Inference — NFE comparison")
    print(f"{'─'*50}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt   = torch.load(checkpoint_path, map_location='cpu')
    config = ckpt.get('config', 'small')
    model  = MiniDiT(n_classes=10, **CONFIGS[config]).to(DEVICE)
    # Use EMA weights
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print(f"[MODEL] Loaded EMA weights  |  epoch={ckpt['epoch']}"
              f"  val={ckpt.get('val_loss','?'):.4f}")
    else:
        model.load_state_dict(ckpt['model'])
        print("[MODEL] Loaded raw weights")
    model.eval()

    # ── Prepare labels ───────────────────────────────────────────────────────
    # Generate n_per_class images for each class (10 classes)
    n_classes  = 10
    n_samples  = n_per_class * n_classes
    y = torch.tensor([c for c in range(n_classes) for _ in range(n_per_class)],
                     device=DEVICE)

    # ── Run all solvers ───────────────────────────────────────────────────────
    configs_solver = [
    ('Euler',  50,  euler_sample, 50),
    ('Euler', 100,  euler_sample, 100),
    ('Euler', 200,  euler_sample, 200),
    ('RK4',    20,  rk4_sample,   80),    # NFE = 4*steps
    ('RK4',    50,  rk4_sample,   200),
]

    results = {}
    for solver, steps, fn, nfe in configs_solver:
        key = f'{solver}-{steps}'
        print(f"[SAMPLE] {key:10s}  NFE={nfe:>3}  ...", end=' ', flush=True)
        t0 = time.time()
        kwargs = {'n_steps': steps, 'guidance': guidance}
        imgs   = fn(model, n_samples, y, **kwargs, device=DEVICE)
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s  ({elapsed/n_samples*1000:.0f}ms/img)")
        results[key] = {'imgs': imgs, 'nfe': nfe, 'solver': solver, 'steps': steps}

    # ── Draw grid: rows=class, cols=solver_config ───────────────────────────────
    _plot_nfe_grid(results, CIFAR10_CLASSES, n_per_class,
                   out_dir / 'nfe_comparison_grid.png')

    # ── Draw loss curve (If have) ─────────────────────────────────────────────────
    log_path = Path(out_dir).parent / 'train_log.json'
    if log_path.exists():
        _plot_loss(log_path, out_dir / 'loss_curve.png')

    print(f"\n[DONE]  Ảnh lưu tại → {out_dir}/")


def _plot_nfe_grid(results, class_names, n_per_class, out_path):
    """
    Layout: rows = solver config, cols = class samples
    Each cell = n_per_class image of same class
    """
    plt.style.use('dark_background')
    solver_keys = list(results.keys())
    n_cols = 10  # 10 classes

    fig, axes = plt.subplots(
        len(solver_keys), n_cols,
        figsize=(n_cols * 1.8, len(solver_keys) * 2.2),
        facecolor='#0a0a0c'
    )
    plt.subplots_adjust(hspace=0.08, wspace=0.04,
                        left=0.10, right=0.99, top=0.93, bottom=0.02)

    for r, key in enumerate(solver_keys):
        data  = results[key]
        imgs  = data['imgs']
        nfe   = data['nfe']

        axes[r, 0].set_ylabel(f"{key}\nNFE={nfe}",
                              color='#00e5ff', fontsize=7.5,
                              fontweight='bold', rotation=0,
                              ha='right', va='center', labelpad=55)

        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_facecolor('#0a0a0c')

            # Take the first image of class c
            idx = c * n_per_class
            ax.imshow(to_img(imgs[idx]))
            ax.axis('off')

            if r == 0:
                ax.set_title(class_names[c], color='white',
                             fontsize=7, pad=3)

    fig.suptitle(
        f'CIFAR-10 Flow Matching  ·  NFE comparison  ·  guidance={4.0}',
        color='white', fontsize=10, fontweight='bold', y=0.97
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0c')
    plt.close(fig)
    print(f"[GRID]  → {out_path}")


def _plot_loss(log_path, out_path):
    with open(log_path, encoding='utf-8') as f:
        log = json.load(f)
    eps    = [d['epoch'] for d in log]
    trains = [d['train'] for d in log]
    vals   = [d['val']   for d in log]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0a0a0c')
    ax.set_facecolor('#13131a')
    ax.plot(eps, trains, color='#00e5ff', lw=1.5, label='train')
    ax.plot(eps, vals,   color='#ff4081', lw=1.5, label='val')
    ax.fill_between(eps, trains, alpha=0.1, color='#00e5ff')
    ax.fill_between(eps, vals,   alpha=0.1, color='#ff4081')
    ax.set_xlabel('Epoch', color='#8899aa')
    ax.set_ylabel('CFM Loss (MSE)', color='#8899aa')
    ax.set_title('Training curve — CIFAR-10 Flow Matching',
                 color='white', fontsize=11)
    ax.legend(fontsize=9, facecolor='#13131a', labelcolor='white')
    ax.tick_params(colors='#667788')
    for sp in ax.spines.values(): sp.set_color('#252530')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0a0a0c')
    plt.close(fig)
    print(f"[LOSS]  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Flow Matching Phase 2 — CIFAR-10 local'
    )
    ap.add_argument('--mode',       default='all',
                    choices=['all', 'precompute', 'train', 'infer'],
                    help='all = run 3 squential steps')
    ap.add_argument('--data_dir',   default='./data',
                    help='Folder download CIFAR-10 ')
    ap.add_argument('--latent_dir', default='outputs/latents')
    ap.add_argument('--out_dir',    default='outputs')
    ap.add_argument('--config',     default='small',
                    choices=['tiny','small','base'],
                    help='tiny=3.5M(CPU) small=14M(RTX) base=33M(RTX3080+)')
    ap.add_argument('--epochs',     type=int,   default=100)
    ap.add_argument('--batch',      type=int,   default=256)
    ap.add_argument('--lr',         type=float, default=1e-4)
    ap.add_argument('--cfg_dropout',type=float, default=0.1)
    ap.add_argument('--guidance',   type=float, default=4.0)
    ap.add_argument('--checkpoint', default='outputs/checkpoints/best.pt')
    args = ap.parse_args()

    print(f"\n{'='*55}")
    print(f"  Flow Matching Phase 2 — CIFAR-10")
    print(f"  mode={args.mode}  config={args.config}  device={DEVICE}")
    print(f"{'='*55}\n")

    if args.mode in ('all', 'precompute'):
        step1_precompute(out_dir=args.latent_dir, data_dir=args.data_dir)

    if args.mode in ('all', 'train'):
        ckpt = step2_train(
            latent_dir   = args.latent_dir,
            out_dir      = args.out_dir,
            config       = args.config,
            epochs       = args.epochs,
            batch        = args.batch,
            lr           = args.lr,
            cfg_dropout  = args.cfg_dropout,
        )
        args.checkpoint = str(ckpt)

    if args.mode in ('all', 'infer'):
        step3_infer(
            checkpoint_path = args.checkpoint,
            out_dir         = Path(args.out_dir) / 'inference',
            guidance        = args.guidance,
        )
