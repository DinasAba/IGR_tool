# Optimizuotas treniravimo ciklas atribucijos atsparumo eksperimentams.
# Pagrindines optimizacijos:
#   1. PGD atakai naudojam CE/KL (be IG) -> ~5x maziau forward'u
#   2. IG(x_clean) skaiciuojam karta ir perpanaudojam
#   3. Konfiguruojamas ig_steps (3 treniravimui, 50 eval)
#   4. AMP (mixed precision) GPU greitinimui
#   5. Konfiguruojamas adv metodas (AT/TRADES/MART) ir reguliarizatorius
# Rezultatas: ~5-8x greiciau per epocha nei originalus IFIA kodas.

import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from .concordance import get_regularizer
from .datasets import get_loaders
from .ig import integrated_gradients
from .losses import ADV_ATTACK_LOSS, LOSSES, pgd_attack
from .models import build_model
from .utils import ensure_dir, get_device, set_seed


# ------------------------------------------------------------------ helperiai
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def _make_save_path(cfg) -> str:
    train_cfg = cfg["train"]

    # jei configas turi kelia - jis pirmumas
    if train_cfg.get("save_path"):
        return train_cfg["save_path"]

    # auto-generavimas pagal eksperimento nustatymus
    dataset = cfg["data"]["name"]
    adv = train_cfg.get("adv_method", "none")
    igr_cfg = cfg.get("igr", {})
    reg = igr_cfg.get("regularizer", "none") if igr_cfg.get("enabled") else "none"
    return f"artifacts/{dataset}_{adv}_{reg}.pt"


def _build_optimizer(model, cfg):
    opt_name = cfg["train"].get("optimizer", "adam").lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.0))

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg["train"].get("momentum", 0.9)),
            weight_decay=wd,
        )
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(opt, cfg):
    sched_type = cfg["train"].get("lr_schedule")
    if sched_type == "step":
        milestones = cfg["train"].get("lr_step_epochs", [75, 90])
        gamma = float(cfg["train"].get("lr_gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma)
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(cfg["train"]["epochs"])
        )
    return None


# ------------------------------------------------------------------ main
def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    dataset = cfg["data"]["name"]
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- duomenys
    train_loader, test_loader = get_loaders(
        cfg["data"]["name"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 2),
    )

    # ---- modelis
    model = build_model(cfg["model"]["name"]).to(device)
    opt = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(opt, cfg)

    # ---- adv metodas
    adv_method = cfg["train"].get("adv_method", "none")
    adv_beta = float(cfg["train"].get("adv_beta", 6.0))
    loss_fn = LOSSES[adv_method]
    attack_loss_type = ADV_ATTACK_LOSS[adv_method]

    attack_cfg = cfg.get("attack", {})
    attack_eps = float(attack_cfg.get("eps", 0.3))
    attack_steps = int(attack_cfg.get("steps", 3))
    attack_step_size = float(attack_cfg.get("step_size", 0.01))

    # ---- atribucijos reguliarizatorius
    igr_cfg = cfg.get("igr", {})
    igr_enabled = bool(igr_cfg.get("enabled", False))
    lambda_igr = float(igr_cfg.get("lambda_igr", 1.0))
    ig_steps = int(igr_cfg.get("ig_steps", 3))
    use_abs_attr = bool(igr_cfg.get("use_abs_attr", False))

    regularizer = None
    if igr_enabled:
        reg_name = igr_cfg.get("regularizer", "cosine")
        reg_kwargs = {}
        if "+" in reg_name:
            # kombinuotas reguliarizatorius, pvz. "cosine+diff_kendall"
            reg_kwargs["w1"] = float(igr_cfg.get("combo_w1", 0.5))
            reg_kwargs["w2"] = float(igr_cfg.get("combo_w2", 0.5))
            # sub-reguliarizatoriu parametrai su r1_/r2_ prefiksu
            reg_kwargs["r2_alpha"] = float(igr_cfg.get("diff_kendall_alpha", 20.0))
            reg_kwargs["r2_n_pairs"] = int(igr_cfg.get("diff_kendall_pairs", 5000))
            reg_kwargs["r2_beta"] = float(igr_cfg.get("soft_spearman_beta", 10.0))
            reg_kwargs["r2_max_dim"] = int(igr_cfg.get("soft_spearman_max_dim", 200))
        elif reg_name == "diff_kendall":
            reg_kwargs["alpha"] = float(igr_cfg.get("diff_kendall_alpha", 20.0))
            reg_kwargs["n_pairs"] = int(igr_cfg.get("diff_kendall_pairs", 5000))
        elif reg_name == "soft_spearman":
            reg_kwargs["beta"] = float(igr_cfg.get("soft_spearman_beta", 10.0))
            reg_kwargs["max_dim"] = int(igr_cfg.get("soft_spearman_max_dim", 200))
        regularizer = get_regularizer(reg_name, **reg_kwargs)

    # ar reikia adv pavyzdziu?
    need_adv = adv_method != "none" or igr_enabled

    # ---- AMP (mixed precision GPU greitinimui)
    use_amp = bool(cfg.get("amp", {}).get("enabled", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- issaugojimo kelias
    save_path = _make_save_path(cfg)
    ensure_dir(os.path.dirname(save_path) or "artifacts")

    # ---- grad clipping (kritinis TRADES/MART su SGD CIFAR-10)
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))

    # ---- LR warmup (stabilizuoja pradzia aukstu lr + adv nuostoliu)
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))

    # ---- IGR warmup: pirmom N epochom isjungiam reguliarizatoriu (kad modelis pirma ismoktu baziniu dalyku)
    igr_warmup = int(igr_cfg.get("igr_warmup", 0))

    # ---- konfiguracijos santrauka
    reg_name_str = igr_cfg.get("regularizer", "none") if igr_enabled else "none"
    print(f"Training: dataset={cfg['data']['name']}  model={cfg['model']['name']}")
    print(f"  adv_method={adv_method}  regularizer={reg_name_str}")
    print(f"  igr_enabled={igr_enabled}  lambda={lambda_igr}  ig_steps={ig_steps}")
    print(f"  attack: eps={attack_eps}  steps={attack_steps}  step_size={attack_step_size}")
    if grad_clip > 0:
        print(f"  grad_clip={grad_clip}")
    if warmup_epochs > 0:
        print(f"  warmup_epochs={warmup_epochs}")
    print(f"  AMP={use_amp}  save_path={save_path}")
    print()

    best_acc = 0.0
    epochs = int(cfg["train"]["epochs"])

    # base lr - warmup'ui
    base_lr = float(cfg["train"]["lr"])

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        # LR warmup: tiesiogiai nuo base_lr/10 iki base_lr
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = base_lr * (0.1 + 0.9 * (epoch - 1) / max(warmup_epochs - 1, 1))
            for pg in opt.param_groups:
                pg["lr"] = warmup_lr

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        ce_sum = reg_sum = total_sum = 0.0
        n_batches = 0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            # ---- 1 zingsnis: adv pavyzdziai per standartini PGD (greita, be IG!)
            if need_adv:
                x_adv = pgd_attack(
                    model, x, y,
                    eps=attack_eps,
                    steps=attack_steps,
                    step_size=attack_step_size,
                    attack_loss=attack_loss_type,
                )
                model.train()   # pgd_attack palieka model.eval()
            else:
                x_adv = x.detach()

            # ---- 2 zingsnis: robustnesio klasifikacinis nuostolis
            with torch.amp.autocast("cuda", enabled=use_amp):
                robust_loss = loss_fn(model, x, x_adv, y, beta=adv_beta)

                # AT/MART treniruojasi tik ant adv - be svaraus signalo.
                # Sudetingesniuose rinkiniuose (OCTMNIST, DermaMNIST, PneumoniaMNIST)
                # tai sukelia kolapsa. Pridejus svaru CE gaunam stabilu signala
                # (kaip TRADES tari natūraliai). MNIST/Fashion-MNIST atveju
                # originalus AT/MART veikia gerai, todel isvengiam nukrypimo nuo
                # Madry / Wang straipsniu.
                needs_clean_ce = dataset not in ("mnist", "fashion_mnist")
                if adv_method in ("at", "mart") and needs_clean_ce:
                    robust_loss = robust_loss + F.cross_entropy(model(x), y)

            # ---- 3 zingsnis: atribucijos reguliarizatorius (pilnas precision stabilumui)
            igr_active = regularizer is not None and lambda_igr > 0 and epoch > igr_warmup
            if igr_active:
                # ig_x = "tikslas" (detach) - be 2 eiles gradientu per svaria saka.
                # Gradientai teka tik per ig_adv - ta ir optimizuojam.
                # Apsauga nuo gradientu konflikto, kuris stebimas AT/MART treniravime.
                ig_x = integrated_gradients(
                    model, x, y, steps=ig_steps, create_graph=False,
                ).detach()
                ig_adv = integrated_gradients(
                    model, x_adv, y, steps=ig_steps, create_graph=True,
                )
                if use_abs_attr:
                    ig_x, ig_adv = ig_x.abs(), ig_adv.abs()

                reg_loss = regularizer(ig_x, ig_adv).mean()
                total_loss = robust_loss + lambda_igr * reg_loss
            else:
                reg_loss = torch.zeros((), device=device)
                total_loss = robust_loss

            # ---- 4 zingsnis: backward + optimizer
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            # ---- statistikos kaupimas
            ce_sum += robust_loss.item()
            reg_sum += reg_loss.item()
            total_sum += total_loss.item()
            n_batches += 1
            pbar.set_postfix(
                ce=f"{robust_loss.item():.3f}",
                reg=f"{reg_loss.item():.3f}",
                tot=f"{total_loss.item():.3f}",
            )

        # scheduler tik po warmup'o (kad nesikertu su warmup lr pakeitimais)
        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()
        elif warmup_epochs > 0 and epoch == warmup_epochs:
            # warmup pabaigoje grazinam base lr - kad scheduler'is pradetu nuo teisingos reiksmes
            for pg in opt.param_groups:
                pg["lr"] = base_lr

        # ---- vertinimas
        acc = accuracy(model, test_loader, device)
        # geriausia saugom tik po IGR warmup'o - kitaip issaugotume modeli, kuris nebuvo reguliarizuotas
        can_save = (igr_warmup == 0) or (epoch > igr_warmup) or (regularizer is None)
        if acc > best_acc and can_save:
            best_acc = acc
            torch.save({"model": model.state_dict(), "cfg": cfg}, save_path)

        n = max(1, n_batches)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}: acc={acc:.4f} best={best_acc:.4f} "
            f"ce={ce_sum/n:.4f} reg={reg_sum/n:.4f} total={total_sum/n:.4f} "
            f"time={elapsed:.1f}s"
        )

    print(f"\nDone. Best accuracy: {best_acc:.4f}")
    print(f"Saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
