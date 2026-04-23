import argparse

import torch
import yaml
from tqdm import tqdm

from .attack_ifia_proxy import ifia_proxy_attack
from .datasets import get_mnist_loaders
from .ig import integrated_gradients
from .metrics import kendall_tau, topk_intersection
from .models import build_model
from .utils import get_device, set_seed


def _resolve_eval_ckpt_path(cfg) -> str:
    train_cfg = cfg["train"]
    igr_cfg = cfg.get("igr", {})
    if bool(igr_cfg.get("enabled", False)):
        if "save_path_igr" in train_cfg:
            return train_cfg["save_path_igr"]
        if train_cfg.get("save_path") == "artifacts/mnist_standard.pt":
            return "artifacts/mnist_igr.pt"
    return train_cfg.get("save_path", "artifacts/mnist_standard.pt")


@torch.no_grad()
def pick_samples(model, loader, device, n_samples: int, only_correct: bool = True):
    xs, ys = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        if only_correct:
            mask = pred == y
            x, y = x[mask], y[mask]
        if x.numel() == 0:
            continue
        xs.append(x)
        ys.append(y)
        if sum(t.size(0) for t in xs) >= n_samples:
            break

    if not xs:
        raise RuntimeError("Nepavyko atrinkti vertinimo pavyzdziu.")

    x_all = torch.cat(xs, dim=0)[:n_samples]
    y_all = torch.cat(ys, dim=0)[:n_samples]
    return x_all, y_all


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()

    _, test_loader = get_mnist_loaders(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    igr_cfg = cfg.get("igr", {})
    ig_cfg = cfg.get("ig", {})
    use_abs_attr = bool(igr_cfg.get("use_abs_attr", False))
    ig_steps_eval = int(igr_cfg.get("ig_steps_eval", ig_cfg.get("steps_eval", 50)))
    ig_steps_attack = int(ig_cfg.get("steps_attack", igr_cfg.get("ig_steps_train", 5)))

    ckpt_path = _resolve_eval_ckpt_path(cfg)
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg["model"]["name"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x, y = pick_samples(
        model,
        test_loader,
        device,
        n_samples=cfg["eval"]["n_samples"],
        only_correct=cfg["eval"]["only_correct"],
    )
    print(f"Eval on samples: {x.size(0)} (only_correct={cfg['eval']['only_correct']})")

    # originalia IG skaiciuojam karta - eval steps
    ig_x_eval = integrated_gradients(model, x, y, steps=ig_steps_eval, create_graph=False)
    if use_abs_attr:
        ig_x_eval = ig_x_eval.abs()

    topk_scores = []
    kendall_scores = []

    # atakuojam su keliais restart'ais ir suvidurkinam metrikas
    adv_list = ifia_proxy_attack(
        model,
        x,
        y,
        eps=cfg["attack"]["eps"],
        steps=cfg["attack"]["steps"],
        step_size=cfg["attack"]["step_size"],
        restarts=cfg["attack"]["restarts"],
        k=cfg["attack"]["k"],
        ig_steps_attack=ig_steps_attack,
        cls_weight=cfg["attack"]["cls_weight"],
        use_abs_attr=use_abs_attr,
    )

    for x_adv in tqdm(adv_list, desc="Compute metrics across restarts"):
        ig_adv_eval = integrated_gradients(model, x_adv, y, steps=ig_steps_eval, create_graph=False)
        if use_abs_attr:
            ig_adv_eval = ig_adv_eval.abs()

        topk = topk_intersection(ig_x_eval, ig_adv_eval, k=cfg["attack"]["k"]).mean().item()
        topk_scores.append(topk)

        taus = []
        for i in range(x.size(0)):
            v1 = ig_x_eval[i].view(-1)
            v2 = ig_adv_eval[i].view(-1)
            taus.append(kendall_tau(v1, v2, sample_pairs=cfg["metrics"]["kendall_pairs"]))
        kendall_scores.append(sum(taus) / len(taus))

    mode_name = "MNIST+IGR-like" if bool(igr_cfg.get("enabled", False)) else "Standard MNIST"
    print(f"\n=== {mode_name} (Table-2-like) ===")
    print(f"top-k intersection (mean over restarts): {sum(topk_scores) / len(topk_scores):.4f}")
    print(f"Kendall tau        (mean over restarts): {sum(kendall_scores) / len(kendall_scores):.4f}")
    # palyginimui: straipsnyje Standard MNIST ~0.3221 ir ~0.0955 (top-k, Kendall)
    print("Note: paper Standard MNIST is around 0.3221 and 0.0955 (top-k, Kendall).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
