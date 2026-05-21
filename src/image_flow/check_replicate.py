import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights

"""" If we only compare pair-wise image or other geometric methods -> failed because of lacking of semantic feature.
    So the best option is a neural network backbone to detect correctly
"""
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def to_uint8_pil_from_minus1_1(x: torch.Tensor) -> Image.Image:
    x = x.float().clamp(-1, 1)
    x = ((x + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def parse_class(class_name: str, class_id: int):
    if class_name is not None:
        c = class_name.strip().lower()
        if c not in CIFAR10_CLASSES:
            raise ValueError(f"Unknown class '{class_name}'. Allowed: {CIFAR10_CLASSES}")
        return CIFAR10_CLASSES.index(c), c
    if class_id is None:
        raise ValueError('Please provide --class_name or --class_id.')
    if not (0 <= class_id < 10):
        raise ValueError('--class_id must be in [0..9].')
    return class_id, CIFAR10_CLASSES[class_id]


def load_feature_backbone(device):
    try:
        # Load ResNet18 (train in ImageNet for classification), remove the final layer to convert it into feature extractor
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        feat_model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
        tf = weights.transforms()
        return feat_model, tf, 'resnet18'
    except Exception as e:
        print(f'[ReplicateCheck] Warning: failed to load pretrained ResNet18 ({e}). Fallback to pixel-space cosine.')
        return None, None, 'pixel'


@torch.no_grad()
def extract_features_from_pil(images, feat_model, preprocess, device, batch_size=64):
    if feat_model is None or preprocess is None:
        feats = []
        to_t = transforms.ToTensor()
        for img in images:
            x = to_t(img).view(-1)
            x = F.normalize(x, dim=0)
            feats.append(x)
        return torch.stack(feats, dim=0)

    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        x = torch.stack([preprocess(img) for img in batch], dim=0).to(device)
        f = feat_model(x).flatten(1)
        f = F.normalize(f, dim=1)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def run_replicate_check(
    generated_tensors: torch.Tensor,
    class_id: int,
    class_name: str,
    latents_dir: Path,
    out_dir: Path,
    topk: int = 5,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_imgs = torch.load(latents_dir / 'train_imgs.pt', map_location='cpu')
    train_labels = torch.load(latents_dir / 'train_labels.pt', map_location='cpu')

    mask = (train_labels == class_id)
    cls_train = train_imgs[mask]
    cls_idx = torch.where(mask)[0] # mask all other labels (only care about same class id)

    if cls_train.shape[0] == 0:
        raise RuntimeError(f'No training samples found for class_id={class_id} ({class_name}).')

    feat_model, preprocess, mode = load_feature_backbone(device)
    print(f'[ReplicateCheck] Feature mode: {mode}')

    gen_pils = [to_uint8_pil_from_minus1_1(generated_tensors[i]) for i in range(generated_tensors.shape[0])]
    train_pils = [to_uint8_pil_from_minus1_1(cls_train[i]) for i in range(cls_train.shape[0])]

    gen_feat = extract_features_from_pil(gen_pils, feat_model, preprocess, device)
    train_feat = extract_features_from_pil(train_pils, feat_model, preprocess, device)

    # cosine similarity matrix: [Ngen, Ntrain]
    sim = gen_feat @ train_feat.T

    report_lines = []
    for gi in range(sim.shape[0]):
        vals, inds = torch.topk(sim[gi], k=min(topk, sim.shape[1]), largest=True)
        gen_dir = out_dir / f'gen_{gi:03d}'
        gen_dir.mkdir(parents=True, exist_ok=True)

        gen_pils[gi].save(gen_dir / f'generated_{class_name}_{gi:03d}.png')

        report_lines.append(f'Generated image {gi:03d}:')
        for rank, (v, local_idx) in enumerate(zip(vals.tolist(), inds.tolist()), start=1):
            global_train_idx = int(cls_idx[local_idx].item())
            near_img = train_pils[local_idx]
            fname = f'top{rank}_nearest_image{global_train_idx}.png'
            near_img.save(gen_dir / fname)
            report_lines.append(f'  top{rank}: train_index={global_train_idx}, cosine={v:.6f}, file={fname}')

    (out_dir / 'replicate_report.txt').write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'[ReplicateCheck] Saved top-{topk} nearest results to: {out_dir}')


def main():
    ap = argparse.ArgumentParser(description='Top-k nearest-neighbor replicate check (generated vs CIFAR-10 train class subset).')
    ap.add_argument('--generated_dir', type=str, required=True,
                    help='Directory containing generated PNG images.')
    ap.add_argument('--class_name', type=str, default=None)
    ap.add_argument('--class_id', type=int, default=None)
    ap.add_argument('--latents_dir', type=str, default='../inference/latents')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--out_dir', type=str, default='../inference/improve_inference/replicate_check')
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    gen_dir = (script_dir / args.generated_dir).resolve() if not Path(args.generated_dir).is_absolute() else Path(args.generated_dir)
    lat_dir = (script_dir / args.latents_dir).resolve() if not Path(args.latents_dir).is_absolute() else Path(args.latents_dir)
    out_dir = (script_dir / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    cid, cname = parse_class(args.class_name, args.class_id)

    img_paths = sorted([p for p in gen_dir.glob('*.png')])
    if not img_paths:
        raise FileNotFoundError(f'No PNG files found in: {gen_dir}')

    # convert generated png -> [-1,1] tensor for shared check function
    to_tensor = transforms.ToTensor()
    imgs = []
    for p in img_paths:
        im = Image.open(p).convert('RGB')
        x = to_tensor(im)
        x = x * 2 - 1
        imgs.append(x)
    gen_t = torch.stack(imgs, dim=0)

    run_replicate_check(gen_t, cid, cname, lat_dir, out_dir, topk=args.topk)


if __name__ == '__main__':
    main()
