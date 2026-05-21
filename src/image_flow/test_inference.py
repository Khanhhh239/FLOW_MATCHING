import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from check_replicate import run_replicate_check

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        print(f"[Device] {props.name} | VRAM={props.total_memory/1e9:.2f} GB")
        return dev
    print("[Device] CPU")
    return torch.device('cpu')


def sinusoidal_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    x = t[:, None] * freqs[None] * 1000
    return torch.cat([x.sin(), x.cos()], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, dim=320):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SDPAAttn(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SDPAAttn(dim, n_heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g1[:, None] * h
        x = x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class MiniDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, dim=320, depth=8, n_heads=8, n_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim) * 0.02)

        self.time_mlp = nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.class_emb = nn.Embedding(n_classes + 1, dim)
        self.null_cls = n_classes

        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_linear = nn.Linear(dim, patch_size * patch_size * in_ch)

    def forward(self, x, t, y=None):
        b = x.shape[0]
        if y is None:
            y = torch.full((b,), self.null_cls, device=x.device, dtype=torch.long)

        tok = self.patch_embed(x) + self.pos_embed
        te = sinusoidal_embed(t, 128)
        c = self.time_mlp(te) + self.class_emb(y)

        for blk in self.blocks:
            tok = blk(tok, c)

        s, sc = self.final_ada(c).chunk(2, dim=-1)
        tok = modulate(self.final_norm(tok), s, sc)
        tok = self.final_linear(tok)

        p = self.patch_size
        h = w = int(self.n_patches ** 0.5)
        tok = tok.reshape(b, h, w, p, p, self.in_ch)
        tok = torch.einsum('bhwpqc->bchpwq', tok)
        return tok.reshape(b, self.in_ch, h * p, w * p)

    @torch.no_grad()
    def forward_cfg(self, x, t, y, guidance=2.0, guidance_rescale=0.7, eps=1e-6):
        null = torch.full_like(y, self.null_cls)
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        y2 = torch.cat([y, null], dim=0)

        u = self.forward(x2, t2, y2)
        u_c, u_u = u.chunk(2, dim=0)
        u_cfg = u_u + guidance * (u_c - u_u)

        std_c = u_c.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1)
        std_cfg = u_cfg.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1)
        u_rescaled = u_cfg * (std_c / (std_cfg + eps))
        return guidance_rescale * u_rescaled + (1.0 - guidance_rescale) * u_cfg


def to_pil_img(x):
    x = x.float().clamp(-1, 1)
    x = ((x + 1.0) / 2.0 * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


@torch.no_grad()
def rk4_sample(model, n, y, n_steps=80, guidance=2.0, guidance_rescale=0.7, device=torch.device('cuda')):
    model.eval()
    x = torch.randn(n, 3, 32, 32, device=device)
    dt = 1.0 / n_steps

    def vf(xt, ti):
        t = torch.full((n,), ti, device=device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
            return model.forward_cfg(xt, t, y, guidance=guidance, guidance_rescale=guidance_rescale)

    for i in range(n_steps):
        ti = i / n_steps
        k1 = vf(x, ti)
        k2 = vf(x + 0.5 * dt * k1, ti + 0.5 * dt)
        k3 = vf(x + 0.5 * dt * k2, ti + 0.5 * dt)
        k4 = vf(x + dt * k3, ti + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def parse_class_input(class_name: str, class_id: int):
    if class_name is not None:
        c = class_name.strip().lower()
        if c not in CIFAR10_CLASSES:
            raise ValueError(f"Unknown class '{class_name}'. Allowed: {CIFAR10_CLASSES}")
        return CIFAR10_CLASSES.index(c)

    if class_id is None:
        return CIFAR10_CLASSES.index('frog')

    if class_id < 0 or class_id >= len(CIFAR10_CLASSES):
        raise ValueError('class_id must be in [0, 9].')

    return class_id


def load_model_from_checkpoint(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')

    cfg_name = ckpt.get('config_name', 'base')
    if cfg_name == 'small':
        kwargs = dict(dim=256, depth=6, n_heads=8)
    else:
        kwargs = dict(dim=320, depth=8, n_heads=8)

    model = MiniDiT(n_classes=10, **kwargs).to(device)

    if 'ema_state' in ckpt:
        state = ckpt['ema_state']
    elif 'ema' in ckpt:
        state = ckpt['ema']
    elif 'model_state' in ckpt:
        state = ckpt['model_state']
    else:
        state = ckpt['model']

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {len(unexpected)}")

    print(f"[Checkpoint] Loaded: {ckpt_path}")
    print(f"[Checkpoint] Epoch: {ckpt.get('epoch', 'N/A')} | Config: {cfg_name}")
    return model


def save_samples(images, out_dir: Path, class_name: str, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(images.shape[0]):
        img = to_pil_img(images[i])
        p = out_dir / f"{prefix}_{class_name}_{i:03d}.png"
        img.save(p)
        paths.append(p)
    return paths


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Generate CIFAR-10 images from trained FM-DiT checkpoint using RK4 sampler.')
    parser.add_argument('--checkpoint', type=str,
                        default='../inference/improve_checkpoints/epoch_0500.pt',
                        help='Path to trained checkpoint (.pt).')
    parser.add_argument('--class_name', type=str, default=None,
                        help='Text class name (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).')
    parser.add_argument('--class_id', type=int, default=None,
                        help='Class id in [0..9]. Used if class_name is not set.')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--rk4_steps', type=int, default=80)
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--guidance_rescale', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='../inference/improve_inference/test_single_class')
    parser.add_argument('--check_replicate', action='store_true',
                        help='Optional: run top-k nearest replicate check against training data.')
    parser.add_argument('--replicate_topk', type=int, default=5)
    parser.add_argument('--latents_dir', type=str, default='../inference/latents')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    ckpt_path = (script_dir / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    cls_id = parse_class_input(args.class_name, args.class_id)
    cls_name = CIFAR10_CLASSES[cls_id]
    print(f"[Class] id={cls_id} name={cls_name}")

    model = load_model_from_checkpoint(ckpt_path, device)

    y = torch.full((args.samples,), cls_id, dtype=torch.long, device=device)
    images = rk4_sample(
        model=model,
        n=args.samples,
        y=y,
        n_steps=args.rk4_steps,
        guidance=args.guidance,
        guidance_rescale=args.guidance_rescale,
        device=device,
    )

    out_dir = (script_dir / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    prefix = f"rk4{args.rk4_steps}_g{str(args.guidance).replace('.', 'p')}"
    paths = save_samples(images, out_dir, cls_name, prefix)

    print('[Done] Saved images:')
    for p in paths:
        print(f"  - {p}")

    if args.check_replicate:
        latents_dir = (script_dir / args.latents_dir).resolve() if not Path(args.latents_dir).is_absolute() else Path(args.latents_dir)
        rep_out = out_dir / f"replicate_check_{cls_name}_{prefix}"
        run_replicate_check(
            generated_tensors=images.detach().cpu(),
            class_id=cls_id,
            class_name=cls_name,
            latents_dir=latents_dir,
            out_dir=rep_out,
            topk=args.replicate_topk,
            device=device,
        )


if __name__ == '__main__':
    main()
