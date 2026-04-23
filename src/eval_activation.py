# Aktyvaciju suderinamumas (Figure 4 is straipsnio).
# Matuoja, kiek neuronu issilaiko ta pacia busena (teigiama/neigiama) tarp x ir x_tilde.
# consistency = P(W^T x > 0 n W^T x_tilde > 0) / sqrt(P(W^T x > 0) * P(W^T x_tilde > 0))

import torch
import torch.nn.functional as F
import yaml

from .datasets import get_loaders
from .losses import pgd_attack
from .models import build_model
from .utils import get_device, set_seed


def _register_hooks(model):
    # forward hooks pre-ReLU aktyvacijoms gauti
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            # ReLU atveju input[0] yra pries-aktyvacija
            activations[name] = input[0].detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(make_hook(name)))

    return activations, hooks


def compute_activation_consistency(model, x_clean, x_adv):
    model.eval()

    # svariu aktyvaciju rinkimas
    act_clean, hooks = _register_hooks(model)
    with torch.no_grad():
        model(x_clean)
    clean_acts = {k: v.clone() for k, v in act_clean.items()}

    # adv aktyvaciju rinkimas (ta pacia dict'a perrasom)
    act_adv, _ = act_clean, hooks
    with torch.no_grad():
        model(x_adv)
    adv_acts = {k: v.clone() for k, v in act_adv.items()}

    for h in hooks:
        h.remove()

    layer_consistency = {}
    for name in clean_acts:
        if name not in adv_acts:
            continue
        a_clean = clean_acts[name].view(clean_acts[name].size(0), -1)
        a_adv = adv_acts[name].view(adv_acts[name].size(0), -1)

        both_pos = ((a_clean > 0) & (a_adv > 0)).float().mean(dim=1)
        p_clean = (a_clean > 0).float().mean(dim=1)
        p_adv = (a_adv > 0).float().mean(dim=1)

        denom = (p_clean * p_adv).sqrt().clamp(min=1e-8)
        consistency = (both_pos / denom).mean().item()
        layer_consistency[name] = consistency

    overall = sum(layer_consistency.values()) / max(len(layer_consistency), 1)
    return {"per_layer": layer_consistency, "overall": overall}


def evaluate_activation(config_path: str, n_samples: int = 200, eps: float = None) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()

    model = build_model(cfg["model"]["name"]).to(device)
    ckpt_path = cfg["train"].get("save_path")
    if not ckpt_path:
        dataset = cfg["data"]["name"]
        adv = cfg["train"].get("adv_method", "none")
        igr_cfg = cfg.get("igr", {})
        reg = igr_cfg.get("regularizer", "none") if igr_cfg.get("enabled") else "none"
        ckpt_path = f"artifacts/{dataset}_{adv}_{reg}.pt"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, test_loader = get_loaders(
        cfg["data"]["name"],
        batch_size=64,
        num_workers=cfg["data"].get("num_workers", 2),
    )

    if eps is None:
        eps = float(cfg.get("attack", {}).get("eps", 0.3))

    # pavyzdziu rinkimas (tik teisingai klasifikuoti)
    xs, ys = [], []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x).argmax(1)
        mask = pred == y
        if mask.any():
            xs.append(x[mask])
            ys.append(y[mask])
        if sum(t.size(0) for t in xs) >= n_samples:
            break

    x_clean = torch.cat(xs)[:n_samples]
    y_clean = torch.cat(ys)[:n_samples]

    # PGD mazesniais paketais - kad CIFAR ResNet-18 neiseitu is atminties
    adv_parts = []
    atk_batch = 32
    for i in range(0, x_clean.size(0), atk_batch):
        xb = x_clean[i:i+atk_batch]
        yb = y_clean[i:i+atk_batch]
        xadv_b = pgd_attack(model, xb, yb, eps=eps, steps=20, step_size=eps/4)
        adv_parts.append(xadv_b)
        torch.cuda.empty_cache()
    x_adv = torch.cat(adv_parts)
    del adv_parts

    result = compute_activation_consistency(model, x_clean, x_adv)

    adv_name = cfg["train"].get("adv_method", "none")
    igr_cfg = cfg.get("igr", {})
    reg_name = igr_cfg.get("regularizer", "none") if igr_cfg.get("enabled") else "none"
    label = f"{adv_name}+{reg_name}" if reg_name != "none" else adv_name

    return {label: result}
