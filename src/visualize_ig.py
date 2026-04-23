# Figure-3 tipo IG vizualizacija: originalus vaizdas | IG(švarus, bazinis) | IG(adv, bazinis) | IG(švarus, IGR) | IG(adv, IGR)
# Normalizacija dalijama pagal modeli (bazinis vs IGR), kad spalvos butu palyginamos.

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from .ig import integrated_gradients
from .models import build_model
from .datasets import get_loaders
from .utils import get_device, set_seed


def _load_model(model_name: str, ckpt_path: str, device: torch.device):
    model = build_model(model_name).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def _pgd_attack_simple(model, x, y, eps, steps=20, step_size=None):
    # paprasta PGD ataka vizualizacijai (vienas restartas)
    if step_size is None:
        step_size = eps / 4
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1).requires_grad_(True)

    for _ in range(steps):
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = x_adv.clamp(0, 1).requires_grad_(True)

    return x_adv.detach()


def _ig_to_absolute_heatmap(ig_attr):
    # |IG| sumuojam per kanalus -> [H, W]. Kaip Wang & Kong vizualizuoja.
    heatmap = ig_attr.abs().sum(dim=0).cpu().numpy()
    return heatmap


def _image_to_np(x, is_color=False):
    if is_color:
        return x.cpu().permute(1, 2, 0).numpy().clip(0, 1)
    else:
        return x.cpu().squeeze(0).numpy()


def generate_ig_visualization(
    dataset: str,
    baseline_ckpt: str,
    igr_ckpt: str,
    model_name: str,
    baseline_label: str = "Baseline",
    igr_label: str = "IGR",
    n_samples: int = 3,
    eps: float = 0.3,
    pgd_steps: int = 20,
    ig_steps: int = 50,
    seed: int = 42,
):
    set_seed(seed)
    device = get_device()
    is_color = dataset in ("cifar10", "dermamnist")

    model_base = _load_model(model_name, baseline_ckpt, device)
    model_igr = _load_model(model_name, igr_ckpt, device)

    _, test_loader = get_loaders(dataset, batch_size=64, num_workers=0)

    # renkam tik teisingai klasifikuojamus abieju modeliu pavyzdzius
    samples = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred_base = model_base(x_batch).argmax(1)
            pred_igr = model_igr(x_batch).argmax(1)
            mask = (pred_base == y_batch) & (pred_igr == y_batch)
            for i in range(mask.sum().item()):
                idx = mask.nonzero(as_tuple=True)[0][i]
                samples.append((x_batch[idx], y_batch[idx]))
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break

    if not samples:
        raise ValueError("No correctly classified samples found")

    FASHION_CLASSES = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    DERMA_CLASSES = [
        "actinic ker.", "basal cell", "benign ker.",
        "dermatofib.", "melanoma", "melanoc. nevi", "vascular",
    ]
    OCT_CLASSES = [
        "CNV", "DME", "DRUSEN", "NORMAL",
    ]

    fig, axes = plt.subplots(
        n_samples, 5,
        figsize=(14, 3.0 * n_samples + 0.8),
        gridspec_kw={"wspace": 0.08, "hspace": 0.35},
    )
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Image",
        f"Original IG\n({baseline_label})",
        f"Adv IG\n({baseline_label})",
        f"Original IG\n({igr_label})",
        f"Adv IG\n({igr_label})",
    ]

    for row, (x, y) in enumerate(samples):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        x_adv_base = _pgd_attack_simple(model_base, x, y, eps, pgd_steps)
        x_adv_igr = _pgd_attack_simple(model_igr, x, y, eps, pgd_steps)

        ig_clean_base = integrated_gradients(model_base, x, y, steps=ig_steps)[0]
        ig_adv_base = integrated_gradients(model_base, x_adv_base, y, steps=ig_steps)[0]
        ig_clean_igr = integrated_gradients(model_igr, x, y, steps=ig_steps)[0]
        ig_adv_igr = integrated_gradients(model_igr, x_adv_igr, y, steps=ig_steps)[0]

        heatmaps = [
            _ig_to_absolute_heatmap(ig_clean_base),
            _ig_to_absolute_heatmap(ig_adv_base),
            _ig_to_absolute_heatmap(ig_clean_igr),
            _ig_to_absolute_heatmap(ig_adv_igr),
        ]

        # atskira vmax baziniam ir IGR modeliui - kitaip stiprus IGR nuplautu silpna bazini
        vmax_base = max(heatmaps[0].max(), heatmaps[1].max(), 1e-8)
        vmax_igr  = max(heatmaps[2].max(), heatmaps[3].max(), 1e-8)
        vmaxes = [vmax_base, vmax_base, vmax_igr, vmax_igr]

        ax = axes[row, 0]
        img_np = _image_to_np(x[0], is_color)
        if is_color:
            ax.imshow(img_np)
        else:
            ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)

        cls_idx = y.item()
        if dataset == "fashion_mnist" and cls_idx < len(FASHION_CLASSES):
            label = FASHION_CLASSES[cls_idx]
        elif dataset == "dermamnist" and cls_idx < len(DERMA_CLASSES):
            label = DERMA_CLASSES[cls_idx]
        elif dataset == "octmnist" and cls_idx < len(OCT_CLASSES):
            label = OCT_CLASSES[cls_idx]
        else:
            label = f"Class {cls_idx}"
        ax.set_ylabel(label, fontsize=11, fontweight="bold")

        for col, heatmap in enumerate(heatmaps):
            ax = axes[row, col + 1]
            ax.imshow(heatmap, cmap="inferno", vmin=0, vmax=vmaxes[col])

        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Attribution Robustness — {dataset.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return fig
