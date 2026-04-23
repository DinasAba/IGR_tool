# White-box adversarinio atsparumo vertinimas (straipsnio 6.3 skyrius).
# Atakos: Natural (svarus), FGSM (Goodfellow 2015), PGD-20 (Madry 2018), CW-Linf (Carlini & Wagner 2017).
# Eps: MNIST/Fashion-MNIST = 0.3, CIFAR-10 = 8/255.

import gc

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from .datasets import get_loaders
from .models import build_model
from .utils import get_device, set_seed


def fgsm_attack(model, x, y, eps):
    # FGSM: vieno zingsnio signo ataka
    x_adv = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = (x + eps * grad.sign()).clamp(0, 1)
    return x_adv.detach()


def pgd_attack(model, x, y, eps, steps=20, step_size=None, restarts=1):
    # PGD su atsitiktiniais restart'ais - grazinam blogiausia atveji per pavyzdi
    if step_size is None:
        step_size = eps / 4
    model.eval()
    best_adv = x.clone()
    best_loss = torch.full((x.size(0),), -float('inf'), device=x.device)

    for _ in range(restarts):
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y, reduction='none')
            loss.sum().backward()
            grad = x_adv.grad.detach()
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)

        with torch.no_grad():
            final_loss = F.cross_entropy(model(x_adv), y, reduction='none')
            improved = final_loss > best_loss
            best_loss[improved] = final_loss[improved]
            best_adv[improved] = x_adv[improved]

    return best_adv.detach()


def cw_inf_attack(model, x, y, eps, steps=30, step_size=None, restarts=1):
    # CW-Linf: PGD su CW margin nuostoliu
    if step_size is None:
        step_size = eps / 4
    model.eval()
    best_adv = x.clone()
    best_loss = torch.full((x.size(0),), -float('inf'), device=x.device)

    for _ in range(restarts):
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            # CW nuostolis: f_wrong - f_correct (maksimizuojam)
            one_hot = F.one_hot(y, logits.size(1)).float()
            correct_logit = (logits * one_hot).sum(1)
            wrong_logit = (logits - 1e4 * one_hot).max(1)[0]
            margin = wrong_logit - correct_logit  # > 0 -> suklasifikuota blogai
            margin.sum().backward()
            grad = x_adv.grad.detach()
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)

        with torch.no_grad():
            logits = model(x_adv)
            one_hot = F.one_hot(y, logits.size(1)).float()
            correct_logit = (logits * one_hot).sum(1)
            wrong_logit = (logits - 1e4 * one_hot).max(1)[0]
            final_margin = wrong_logit - correct_logit
            improved = final_margin > best_loss
            best_loss[improved] = final_margin[improved]
            best_adv[improved] = x_adv[improved]

    return best_adv.detach()


def eval_adversarial_accuracy(model, loader, device, eps, batch_limit=None):
    model.eval()
    counts = {"natural": 0, "fgsm": 0, "pgd20": 0, "cw_inf": 0}
    total = 0

    pbar = tqdm(loader, desc="Adv accuracy eval")
    for i, (x, y) in enumerate(pbar):
        if batch_limit and i >= batch_limit:
            break
        x, y = x.to(device), y.to(device)
        bs = y.size(0)
        total += bs

        with torch.no_grad():
            counts["natural"] += (model(x).argmax(1) == y).sum().item()

        x_fgsm = fgsm_attack(model, x, y, eps)
        with torch.no_grad():
            counts["fgsm"] += (model(x_fgsm).argmax(1) == y).sum().item()
        del x_fgsm

        x_pgd = pgd_attack(model, x, y, eps, steps=20, restarts=1)
        with torch.no_grad():
            counts["pgd20"] += (model(x_pgd).argmax(1) == y).sum().item()
        del x_pgd

        x_cw = cw_inf_attack(model, x, y, eps, steps=30, restarts=1)
        with torch.no_grad():
            counts["cw_inf"] += (model(x_cw).argmax(1) == y).sum().item()
        del x_cw

        torch.cuda.empty_cache()

        nat_acc = counts["natural"] / total
        pgd_acc = counts["pgd20"] / total
        pbar.set_postfix({"nat": f"{nat_acc:.2%}", "pgd20": f"{pgd_acc:.2%}"})

    results = {k: v / total for k, v in counts.items()}
    return results


def evaluate_adv(config_path: str, batch_limit: int = None) -> dict:
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

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, test_loader = get_loaders(
        cfg["data"]["name"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 2),
    )

    eps = float(cfg.get("attack", {}).get("eps", 0.3))
    print(f"White-box eval: eps={eps}  attacks=[FGSM, PGD-20, CW-inf]")
    results = eval_adversarial_accuracy(model, test_loader, device, eps, batch_limit)

    print(f"\n  Natural:  {results['natural']:.2%}")
    print(f"  FGSM:     {results['fgsm']:.2%}")
    print(f"  PGD-20:   {results['pgd20']:.2%}")
    print(f"  CW-inf:   {results['cw_inf']:.2%}")

    adv_name = cfg["train"].get("adv_method", "none")
    igr_cfg = cfg.get("igr", {})
    reg_name = igr_cfg.get("regularizer", "none") if igr_cfg.get("enabled") else "none"
    label = f"{adv_name}+{reg_name}" if reg_name != "none" else adv_name

    del model, ckpt
    gc.collect()
    torch.cuda.empty_cache()

    return {label: results}
