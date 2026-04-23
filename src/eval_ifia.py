# Vertinimo pipeline su TRUE IFIA(top-k) ataka.
# Kiekvienam checkpoint'ui:
#   1. Ikraunam modeli, atrenkam teisingai klasifikuotus test pavyzdzius
#   2. Paleidziam IFIA ataka (soft-topk arba cosine proxy, konfiguruojama)
#   3. Skaiciuojam tikslias metrikas: top-k, Kendall tau, Spearman rho, cosine, Pearson
#   4. Iseksportuojam rezultatus (CSV / LaTeX / JSON)

import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from .concordance import (
    cosine_sim_batch,
    cosine_sim_exact,
    diff_kendall_sim_batch,
    kendall_tau_exact,
    pearson_exact,
    pearson_sim_batch,
    soft_spearman_sim_batch,
    soft_topk_intersection,
    spearman_rho_exact,
    topk_intersection_exact,
)
from .datasets import get_loaders
from .ig import integrated_gradients
from .models import build_model
from .utils import get_device, set_seed


# =================================================================
#  IFIA ataka (Iterative Feature Importance Attack)
# =================================================================

def ifia_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    ig_clean: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    restarts: int,
    k: int,
    ig_steps: int,
    proxy: str = "soft_topk",
    cls_weight: float = 1.0,
    use_abs_attr: bool = False,
) -> list[torch.Tensor]:
    # IFIA ataka su konfiguruojamu diferencijuojamu surogatu.
    # Proxy variantai:
    #   "soft_topk"     - soft top-k intersection (TRUE IFIA, jautrus k)
    #   "cosine"        - cos panasumas (originalus IGR straipsnio proxy)
    #   "pearson"       - Pearson koreliacija
    #   "diff_kendall"  - diferencijuojamas Kendall (tanh aprox.)
    #   "soft_spearman" - soft Spearman (sigmoid rangavimas)
    model.eval()
    adv_list = []

    total_iters = restarts * steps
    for r in range(restarts):
        x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1)

        pbar = tqdm(range(steps), desc=f"    restart {r+1}/{restarts}",
                    leave=False, ncols=80)
        for s in pbar:
            x_adv.requires_grad_(True)

            ig_adv = integrated_gradients(
                model, x_adv, y, steps=ig_steps, create_graph=False,
            )
            if use_abs_attr:
                ig_adv = ig_adv.abs()

            # diferencijuojamas atribucijos nesutapimo surogatas
            if proxy == "soft_topk":
                sim = soft_topk_intersection(ig_clean, ig_adv, k=k, temperature=0.1)
            elif proxy == "cosine":
                sim = cosine_sim_batch(ig_clean, ig_adv)
            elif proxy == "pearson":
                sim = pearson_sim_batch(ig_clean, ig_adv)
            elif proxy == "diff_kendall":
                sim = diff_kendall_sim_batch(ig_clean, ig_adv)
            elif proxy == "soft_spearman":
                sim = soft_spearman_sim_batch(ig_clean, ig_adv)
            else:
                raise ValueError(f"Unknown proxy: {proxy}")

            dissim = (1.0 - sim).mean()

            # reikia islaikyti teisinga klasifikacija
            logits = model(x_adv)
            ce = F.cross_entropy(logits, y)

            # maksimizuojam nesutapima, minimizuojam klasifikacijos klaida
            obj = dissim - cls_weight * ce
            grad = torch.autograd.grad(obj, x_adv)[0]

            with torch.no_grad():
                x_adv = x_adv + step_size * grad.sign()
                x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
                x_adv = x_adv.clamp(0, 1)

        adv_list.append(x_adv.detach())

    return adv_list


# =================================================================
#  Pavyzdziu atranka
# =================================================================

@torch.no_grad()
def pick_correct_samples(model, loader, device, n_samples: int):
    # tik teisingai klasifikuoti
    xs, ys = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        mask = pred == y
        if mask.any():
            xs.append(x[mask])
            ys.append(y[mask])
        if sum(t.size(0) for t in xs) >= n_samples:
            break
    if not xs:
        raise RuntimeError("Nerasta teisingai klasifikuotu pavyzdziu!")
    return torch.cat(xs)[:n_samples], torch.cat(ys)[:n_samples]


# =================================================================
#  Metriku skaiciavimas
# =================================================================

def compute_all_metrics(
    ig_clean: torch.Tensor,
    ig_adv: torch.Tensor,
    k: int,
    kendall_pairs: int = 10000,
) -> dict:
    # visos tikslios metrikos tarp svariu ir adv IG
    n = ig_clean.size(0)
    topk_vals, kt_vals, sp_vals, cos_vals, pear_vals = [], [], [], [], []

    for i in range(n):
        v1 = ig_clean[i].view(-1)
        v2 = ig_adv[i].view(-1)

        topk_vals.append(topk_intersection_exact(v1, v2, k))
        kt_vals.append(kendall_tau_exact(v1, v2, sample_pairs=kendall_pairs))
        sp_vals.append(spearman_rho_exact(v1, v2))
        cos_vals.append(cosine_sim_exact(v1, v2))
        pear_vals.append(pearson_exact(v1, v2))

    def _mean(lst):
        return sum(lst) / len(lst)

    return {
        "top_k": _mean(topk_vals),
        "kendall_tau": _mean(kt_vals),
        "spearman_rho": _mean(sp_vals),
        "cosine_sim": _mean(cos_vals),
        "pearson": _mean(pear_vals),
    }


# =================================================================
#  Eksportas
# =================================================================

def format_latex_row(model_name: str, metrics: dict) -> str:
    vals = " & ".join(f"{metrics[k]:.4f}" for k in
                      ["top_k", "kendall_tau", "spearman_rho", "cosine_sim", "pearson"])
    return f"{model_name} & {vals} \\\\"


def export_results(results: dict, export_dir: str):
    os.makedirs(export_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(export_dir, "results.csv")
    fieldnames = ["model", "top_k", "kendall_tau", "spearman_rho", "cosine_sim", "pearson"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in results.items():
            row = {"model": name, **metrics}
            writer.writerow(row)
    print(f"CSV saved: {csv_path}")

    # LaTeX
    tex_path = os.path.join(export_dir, "results.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "c" * 5 + "}\n")
        f.write("\\toprule\n")
        f.write("Model & Top-k & Kendall $\\tau$ & Spearman $\\rho$ "
                "& Cosine & Pearson \\\\\n")
        f.write("\\midrule\n")
        for name, metrics in results.items():
            f.write(format_latex_row(name, metrics) + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"LaTeX saved: {tex_path}")

    # JSON (programaviniam naudojimui)
    json_path = os.path.join(export_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {json_path}")


# =================================================================
#  Main
# =================================================================

def evaluate(config_path: str, export_dir: str | None = None) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    # ---- Modelis
    model = build_model(cfg["model"]["name"]).to(device)

    # checkpoint kelio nustatymas
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

    # ---- Duomenys
    _, test_loader = get_loaders(
        cfg["data"]["name"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 2),
    )

    # ---- Eval config
    eval_cfg = cfg.get("eval", {})
    n_samples = int(eval_cfg.get("n_samples", 200))

    eval_attack = eval_cfg.get("attack", cfg.get("attack", {}))
    eps = float(eval_attack.get("eps", 0.3))
    steps = int(eval_attack.get("steps", 200))
    step_size = float(eval_attack.get("step_size", 0.01))
    restarts = int(eval_attack.get("restarts", 5))
    k = int(eval_attack.get("k", 100))
    cls_weight = float(eval_attack.get("cls_weight", 1.0))
    proxy = str(eval_attack.get("proxy", "cosine"))

    igr_cfg = cfg.get("igr", {})
    ig_steps_eval = int(eval_cfg.get("ig_steps", igr_cfg.get("ig_steps_eval", 50)))
    use_abs_attr = bool(igr_cfg.get("use_abs_attr", False))
    kendall_pairs = int(cfg.get("metrics", {}).get("kendall_pairs", 10000))

    # ---- Pavyzdziu atranka
    x, y = pick_correct_samples(model, test_loader, device, n_samples)
    print(f"Evaluating on {x.size(0)} correctly classified samples")

    # ---- eval mini-batch dydis (kad neuzsikimstu atmintis dideliuose modeliuose)
    # MNIST CNN mazas; CIFAR-10 ResNet-18 reikia mazesniu paketu
    is_cifar = cfg["data"]["name"].startswith("cifar")
    eval_batch = 10 if is_cifar else 50

    # ---- svariu IG (karta, dideliu precisionu, mini-batch'ais)
    print(f"Computing clean IG (steps={ig_steps_eval}, batch={eval_batch})...")
    ig_clean_parts = []
    for i in range(0, x.size(0), eval_batch):
        xb = x[i:i+eval_batch]
        yb = y[i:i+eval_batch]
        ig_part = integrated_gradients(
            model, xb, yb, steps=ig_steps_eval, create_graph=False,
        )
        ig_clean_parts.append(ig_part.cpu())
        del ig_part
        torch.cuda.empty_cache()
    ig_clean_cpu = torch.cat(ig_clean_parts)
    del ig_clean_parts
    if use_abs_attr:
        ig_clean_cpu = ig_clean_cpu.abs()

    # ---- IFIA ataka (mini-batch'ais)
    ig_steps_attack = int(eval_attack.get("ig_steps_attack", 5))
    print(f"Running IFIA attack: proxy={proxy} steps={steps} restarts={restarts} "
          f"k={k} eps={eps} batch={eval_batch}")
    t0 = time.time()

    # atakuojam paketais - kad neprisigamintu OOM
    n_batches = (x.size(0) + eval_batch - 1) // eval_batch
    adv_list_full = [[] for _ in range(restarts)]
    for batch_idx, i in enumerate(range(0, x.size(0), eval_batch)):
        xb = x[i:i+eval_batch]
        yb = y[i:i+eval_batch]
        # atakai reikia svaraus IG GPU'iu
        ig_clean_b = ig_clean_cpu[i:i+eval_batch].to(device)
        if use_abs_attr:
            ig_clean_b = ig_clean_b.abs()

        print(f"  IFIA batch {batch_idx+1}/{n_batches} "
              f"(samples {i+1}-{min(i+eval_batch, x.size(0))})")

        adv_batch = ifia_attack(
            model, xb, yb, ig_clean_b,
            eps=eps, steps=steps, step_size=step_size,
            restarts=restarts, k=k, ig_steps=ig_steps_attack,
            proxy=proxy, cls_weight=cls_weight, use_abs_attr=use_abs_attr,
        )
        for r_idx, xadv_b in enumerate(adv_batch):
            adv_list_full[r_idx].append(xadv_b.cpu())
        del ig_clean_b, adv_batch
        torch.cuda.empty_cache()

    # surenkam pilnus adv tenzorius
    adv_list = [torch.cat(parts) for parts in adv_list_full]
    del adv_list_full

    attack_time = time.time() - t0
    print(f"Attack done in {attack_time:.1f}s")

    # ---- metrikos kiekvienam restart'ui, paskui vidurkis
    all_metrics = []
    for r_idx, x_adv_full in enumerate(tqdm(adv_list, desc="Computing metrics per restart")):
        # adv IG mini-batch'ais
        ig_adv_parts = []
        for i in range(0, x_adv_full.size(0), eval_batch):
            xadv_b = x_adv_full[i:i+eval_batch].to(device)
            yb = y[i:i+eval_batch]
            ig_part = integrated_gradients(
                model, xadv_b, yb, steps=ig_steps_eval, create_graph=False,
            )
            ig_adv_parts.append(ig_part.cpu())
            del ig_part, xadv_b
            torch.cuda.empty_cache()
        ig_adv_cpu = torch.cat(ig_adv_parts)
        del ig_adv_parts
        if use_abs_attr:
            ig_adv_cpu = ig_adv_cpu.abs()

        m = compute_all_metrics(ig_clean_cpu, ig_adv_cpu, k=k, kendall_pairs=kendall_pairs)
        all_metrics.append(m)
        del ig_adv_cpu

    # vidurkis per restart'us
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    # ---- rezultatu spausdinimas
    adv_name = cfg["train"].get("adv_method", "none")
    reg_name = igr_cfg.get("regularizer", "none") if igr_cfg.get("enabled") else "none"
    model_label = f"{adv_name}+{reg_name}" if reg_name != "none" else adv_name

    print(f"\n{'='*60}")
    print(f"  Model: {model_label}  ({cfg['data']['name']})")
    print(f"{'='*60}")
    print(f"  Top-k intersection:  {avg_metrics['top_k']:.4f}")
    print(f"  Kendall tau:         {avg_metrics['kendall_tau']:.4f}")
    print(f"  Spearman rho:        {avg_metrics['spearman_rho']:.4f}")
    print(f"  Cosine similarity:   {avg_metrics['cosine_sim']:.4f}")
    print(f"  Pearson:             {avg_metrics['pearson']:.4f}")
    print(f"{'='*60}\n")

    # ---- eksportas
    if export_dir:
        results = {model_label: avg_metrics}
        export_results(results, export_dir)

    return {model_label: avg_metrics}


CROSS_PROXIES = ["cosine", "diff_kendall", "soft_spearman", "pearson"]


def cross_proxy_evaluate(config_path: str, proxies: list[str] | None = None) -> dict:
    # Vertinam viena modeli su keliais IFIA atakos proxy.
    # Grazina {proxy_name: {metric: float}} - viena cross-proxy matricos eilute.
    # Svariu IG ir pavyzdziu atranka daroma karta, perpanaudojam visiems proxy.
    if proxies is None:
        proxies = CROSS_PROXIES

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()

    # ---- Modelis
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

    # ---- Duomenys
    _, test_loader = get_loaders(
        cfg["data"]["name"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 2),
    )

    # ---- Eval config
    eval_cfg = cfg.get("eval", {})
    n_samples = int(eval_cfg.get("n_samples", 200))
    eval_attack = eval_cfg.get("attack", cfg.get("attack", {}))
    eps = float(eval_attack.get("eps", 0.3))
    steps = int(eval_attack.get("steps", 200))
    step_size = float(eval_attack.get("step_size", 0.01))
    restarts = int(eval_attack.get("restarts", 5))
    k = int(eval_attack.get("k", 100))
    cls_weight = float(eval_attack.get("cls_weight", 1.0))
    ig_steps_attack = int(eval_attack.get("ig_steps_attack", 5))

    igr_cfg = cfg.get("igr", {})
    ig_steps_eval = int(eval_cfg.get("ig_steps", igr_cfg.get("ig_steps_eval", 50)))
    use_abs_attr = bool(igr_cfg.get("use_abs_attr", False))
    kendall_pairs = int(cfg.get("metrics", {}).get("kendall_pairs", 10000))

    is_cifar = cfg["data"]["name"].startswith("cifar")
    eval_batch = 10 if is_cifar else 50

    # ---- pavyzdziu atranka (karta)
    x, y = pick_correct_samples(model, test_loader, device, n_samples)

    # ---- svariu IG (karta)
    ig_clean_parts = []
    for i in range(0, x.size(0), eval_batch):
        ig_part = integrated_gradients(
            model, x[i:i+eval_batch], y[i:i+eval_batch],
            steps=ig_steps_eval, create_graph=False,
        )
        ig_clean_parts.append(ig_part.cpu())
        del ig_part; torch.cuda.empty_cache()
    ig_clean_cpu = torch.cat(ig_clean_parts)
    del ig_clean_parts
    if use_abs_attr:
        ig_clean_cpu = ig_clean_cpu.abs()

    # ---- atakuojam kiekvienu proxy
    proxy_results = {}
    for proxy in proxies:
        print(f"\n  --- Cross-proxy attack: {proxy} ---")
        t0 = time.time()

        n_batches = (x.size(0) + eval_batch - 1) // eval_batch
        adv_list_full = [[] for _ in range(restarts)]
        for batch_idx, bi in enumerate(range(0, x.size(0), eval_batch)):
            xb = x[bi:bi+eval_batch]
            yb = y[bi:bi+eval_batch]
            ig_clean_b = ig_clean_cpu[bi:bi+eval_batch].to(device)

            adv_batch = ifia_attack(
                model, xb, yb, ig_clean_b,
                eps=eps, steps=steps, step_size=step_size,
                restarts=restarts, k=k, ig_steps=ig_steps_attack,
                proxy=proxy, cls_weight=cls_weight, use_abs_attr=use_abs_attr,
            )
            for r_idx, xadv_b in enumerate(adv_batch):
                adv_list_full[r_idx].append(xadv_b.cpu())
            del ig_clean_b, adv_batch; torch.cuda.empty_cache()

        adv_list = [torch.cat(parts) for parts in adv_list_full]
        del adv_list_full
        print(f"  Attack done in {time.time()-t0:.1f}s")

        # metrikos
        all_metrics = []
        for x_adv_full in adv_list:
            ig_adv_parts = []
            for i in range(0, x_adv_full.size(0), eval_batch):
                xadv_b = x_adv_full[i:i+eval_batch].to(device)
                ig_part = integrated_gradients(
                    model, xadv_b, y[i:i+eval_batch],
                    steps=ig_steps_eval, create_graph=False,
                )
                ig_adv_parts.append(ig_part.cpu())
                del ig_part, xadv_b; torch.cuda.empty_cache()
            ig_adv_cpu = torch.cat(ig_adv_parts)
            del ig_adv_parts
            if use_abs_attr:
                ig_adv_cpu = ig_adv_cpu.abs()
            m = compute_all_metrics(ig_clean_cpu, ig_adv_cpu, k=k, kendall_pairs=kendall_pairs)
            all_metrics.append(m)
            del ig_adv_cpu

        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        proxy_results[proxy] = avg_metrics

        del adv_list
        torch.cuda.empty_cache()

    return proxy_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--export", default=None, help="Export dir for CSV/LaTeX/JSON")
    args = ap.parse_args()
    evaluate(args.config, args.export)
