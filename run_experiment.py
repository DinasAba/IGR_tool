#!/usr/bin/env python
# CLI konkordacijos matu palyginimo eksperimentams.
# Pavyzdziai:
#   python run_experiment.py --dataset mnist --adv at --reg cosine --epochs 10
#   python run_experiment.py --dataset mnist --run-all --epochs 10
#   python run_experiment.py --dataset mnist --adv at --reg cosine --eval-only

import argparse
import gc
import json
import os
import sys
import tempfile
import time

import torch
import yaml

# kad veiktu src.* importai
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import main as train_main
from src.eval_ifia import evaluate, export_results, cross_proxy_evaluate, CROSS_PROXIES
from src.eval_adversarial import evaluate_adv
from src.eval_activation import evaluate_activation


# Numatyti hyperparametrai kiekvienam rinkiniui (pagal Wang & Kong, NeurIPS 2022)
DATASET_DEFAULTS = {
    "mnist": {
        "model": "mnist_cnn",
        "batch_size": 64,
        "epochs": 90,            # straipsnyje: 90 epochu
        "lr": 1e-4,
        "optimizer": "adam",
        "attack_eps": 0.3,
        "attack_steps_train": 7,            # straipsnyje: 7 PGD zingsniai
        "attack_step_size_train": 0.075,    # straipsnyje: eps/4 = 0.3/4
        "eval_steps": 200,
        "eval_step_size": 0.01,
        "eval_restarts": 5,
        "eval_k": 100,
        "ig_steps_train": 3,
        "ig_steps_eval": 50,
    },
    "fashion_mnist": {
        "model": "mnist_cnn",
        "batch_size": 64,
        "epochs": 90,            # kaip MNIST
        "lr": 1e-4,
        "optimizer": "adam",
        "attack_eps": 0.3,
        "attack_steps_train": 7,
        "attack_step_size_train": 0.075,
        "eval_steps": 200,
        "eval_step_size": 0.01,
        "eval_restarts": 5,
        "eval_k": 100,
        "ig_steps_train": 3,
        "ig_steps_eval": 50,
    },
    "cifar10": {
        "model": "cifar10_resnet18",
        "batch_size": 64,
        "epochs": 120,
        "lr": 0.1,
        "optimizer": "sgd",
        "lr_schedule": "step",
        "lr_step_epochs": [75, 90],
        "lr_gamma": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "grad_clip": 1.0,           # apsauga nuo grad sprogimo (TRADES/MART + SGD)
        "warmup_epochs": 5,         # tiesiogine LR warmup: lr/10 -> lr per 5 epochas
        "attack_eps": 8.0 / 255,
        "attack_steps_train": 7,             # straipsnyje: 7 PGD zingsniai
        "attack_step_size_train": 2.0 / 255, # straipsnyje: eps/4 = 8/255/4 = 2/255
        "eval_steps": 200,
        "eval_step_size": 1.0 / 255,
        "eval_restarts": 5,
        "eval_k": 100,
        "ig_steps_train": 1,           # 1 zingsnis IG ResNet-18 (3 per letas)
        "ig_steps_eval": 30,
        "amp_enabled": True,            # mixed-precision RTX 3070
    },
    "dermamnist": {
        "model": "dermamnist_resnet18",
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-3,              # Adam - stabilus maziems rinkiniams
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "attack_eps": 8.0 / 255,          # kaip CIFAR-10 (RGB, 28x28)
        "attack_steps_train": 7,
        "attack_step_size_train": 2.0 / 255,  # eps/4
        "eval_steps": 200,
        "eval_step_size": 1.0 / 255,
        "eval_restarts": 5,
        "eval_k": 100,
        "ig_steps_train": 3,             # tikras IG (ne Input x Gradient)
        "ig_steps_eval": 30,
        "amp_enabled": True,
        "igr_warmup": 3,                 # pradzioj be IGR - kad modelis ismoktu baziniu dalyku
    },
    "octmnist": {
        "model": "octmnist_cnn",          # grayscale 28x28, 4 klases - tas pats CNN
        "batch_size": 64,
        "epochs": 50,
        "lr": 5e-4,                      # pakankamai mokytis, nepersukti stabilumo
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "attack_eps": 0.05,              # ~13/255, arciau CIFAR-10 skales; 0.1 -> AT kolapsas
        "attack_steps_train": 7,
        "attack_step_size_train": 0.0125, # eps/4
        "eval_steps": 200,
        "eval_step_size": 0.0025,         # eps/20 - tikslesnis eval
        "eval_restarts": 5,
        "eval_k": 100,
        "ig_steps_train": 3,             # tikras IG
        "ig_steps_eval": 50,
        "igr_warmup": 5,                 # 5 epochos be IGR is pradziu
    },
    "pneumoniamnist": {
        "model": "pneumoniamnist_cnn",    # grayscale 28x28, 2 klases
        "batch_size": 64,
        "epochs": 20,                     # ~4.7k train, greitai konverguoja
        "lr": 5e-4,
        "optimizer": "adam",
        "weight_decay": 5e-4,             # stipresnis L2 maziam rinkiniui
        "grad_clip": 1.0,
        "attack_eps": 0.075,              # 0.05 per lengva, 0.1 kolapsas
        "attack_steps_train": 7,
        "attack_step_size_train": 0.01875, # eps/4
        "eval_steps": 200,
        "eval_step_size": 0.00375,        # eps/20
        "eval_restarts": 5,
        "eval_k": 100,
        "ig_steps_train": 3,
        "ig_steps_eval": 30,
        "igr_warmup": 3,                  # 3 epochu warmup pries IGR
    },
}

# Lambda IGR reguliarizatoriui (straipsnyje nenurodyta; default 1.0)
# perrasoma per --lambda-igr
LAMBDA_IGR_DEFAULTS = {
    "at":     1.0,
    "trades": 1.0,
    "mart":   1.0,
}

# Visos kombinacijos palyginimo lentelei
ADV_METHODS = ["none", "at", "trades", "mart"]
REGULARIZERS = ["none", "cosine", "diff_kendall", "soft_spearman", "pearson"]
COMBO_REGULARIZERS = ["cosine+diff_kendall", "cosine+soft_spearman"]


def make_config(dataset, adv_method, reg, epochs=None, seed=123,
                lambda_igr=None, fast_train=False,
                combo_w1=0.5, combo_w2=0.5):
    # config dict vienam eksperimentui
    d = DATASET_DEFAULTS[dataset]
    ep = epochs or d["epochs"]

    # PGD treniravimo parametrai: straipsnio default arba fast
    if fast_train:
        pgd_steps = 3
        pgd_step_size = d["attack_eps"] / 4  # eps-proporcingas, ne fiksuotas 0.01
    else:
        pgd_steps = d["attack_steps_train"]
        pgd_step_size = d["attack_step_size_train"]

    # save path: + -> _ kad tiktu failu sistemai
    reg_safe = reg.replace("+", "_") if reg else reg

    cfg = {
        "seed": seed,
        "data": {
            "name": dataset,
            "batch_size": d["batch_size"],
            "num_workers": 2,
        },
        "model": {"name": d["model"]},
        "train": {
            "epochs": ep,
            "lr": d["lr"],
            "optimizer": d.get("optimizer", "adam"),
            "weight_decay": d.get("weight_decay", 0.0),
            "adv_method": adv_method,
            "adv_beta": 6.0,
            "grad_clip": d.get("grad_clip", 0.0),
            "warmup_epochs": d.get("warmup_epochs", 0),
            "save_path": f"artifacts/{dataset}_{adv_method}_{reg_safe}.pt",
        },
        "attack": {
            "eps": d["attack_eps"],
            "steps": pgd_steps,
            "step_size": pgd_step_size,
        },
        "igr": {
            "enabled": reg != "none",
            "regularizer": reg if reg != "none" else "cosine",
            "lambda_igr": lambda_igr if lambda_igr is not None else LAMBDA_IGR_DEFAULTS.get(adv_method, 1.0),
            "ig_steps": d["ig_steps_train"],
            "use_abs_attr": False,
            "diff_kendall_alpha": 20.0,
            "diff_kendall_pairs": 5000,
            "soft_spearman_beta": 10.0,
            "soft_spearman_max_dim": 200,
            "combo_w1": combo_w1,
            "combo_w2": combo_w2,
            "igr_warmup": d.get("igr_warmup", 0),
        },
        "eval": {
            "n_samples": d.get("eval_n_samples", 200),
            "only_correct": True,
            "ig_steps": d["ig_steps_eval"],
            "attack": {
                "eps": d["attack_eps"],
                "steps": d["eval_steps"],
                "step_size": d["eval_step_size"],
                "restarts": d["eval_restarts"],
                "k": d["eval_k"],
                "cls_weight": 1.0,
                "proxy": "cosine",
                "ig_steps_attack": 5,
            },
        },
        "metrics": {"kendall_pairs": 10000},
        "amp": {"enabled": d.get("amp_enabled", False)},
    }

    # SGD specifiniai
    if d.get("optimizer") == "sgd":
        cfg["train"]["momentum"] = d.get("momentum", 0.9)
        if d.get("lr_schedule"):
            cfg["train"]["lr_schedule"] = d["lr_schedule"]
            cfg["train"]["lr_step_epochs"] = d.get("lr_step_epochs", [75, 90])
            cfg["train"]["lr_gamma"] = d.get("lr_gamma", 0.1)

    return cfg


def save_temp_config(cfg):
    # issaugom i laikina YAML ir grazinam kelia
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="igr_cfg_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


def run_single(dataset, adv, reg, epochs=None, eval_only=False, seed=123,
               lambda_igr=None, fast_train=False,
               do_eval_adv=False, do_eval_activation=False, quick=False,
               combo_w1=0.5, combo_w2=0.5):
    # treniravimas + vertinimas vienam eksperimentui. Grazina (label, metrics_dict, extra).
    label = f"{adv}+{reg}" if reg != "none" else adv
    print(f"\n{'#'*70}")
    print(f"  Experiment: {dataset} / {label}")
    print(f"{'#'*70}\n")

    cfg = make_config(dataset, adv, reg, epochs=epochs, seed=seed,
                      lambda_igr=lambda_igr, fast_train=fast_train,
                      combo_w1=combo_w1, combo_w2=combo_w2)
    cfg_path = save_temp_config(cfg)

    # taip pat issaugom pastovia kopija - kad butu istorija
    perm_dir = os.path.join("configs", "generated")
    os.makedirs(perm_dir, exist_ok=True)
    perm_path = os.path.join(perm_dir, f"{dataset}_{adv}_{reg}.yaml")
    with open(perm_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    extra = {}
    try:
        if not eval_only:
            train_main(cfg_path)

        results = evaluate(cfg_path)
        metrics = list(results.values())[0]

        if do_eval_adv:
            batch_lim = 10 if quick else None
            adv_res = evaluate_adv(cfg_path, batch_limit=batch_lim)
            extra["adv_accuracy"] = list(adv_res.values())[0]

        if do_eval_activation:
            n = 50 if quick else 200
            act_res = evaluate_activation(cfg_path, n_samples=n)
            extra["activation"] = list(act_res.values())[0]

        return label, metrics, extra
    finally:
        os.unlink(cfg_path)


def get_valid_combinations():
    # visos (adv_method, regularizer) kombinacijos
    combos = []
    for adv in ADV_METHODS:
        # standartinis (be adv, be reg)
        if adv == "none":
            combos.append(("none", "none"))
            continue
        # adv be reg
        combos.append((adv, "none"))
        # adv su kiekvienu reg
        for reg in REGULARIZERS:
            if reg != "none":
                combos.append((adv, reg))
    return combos


def main():
    ap = argparse.ArgumentParser(description="Concordance measure comparison framework")
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS.keys()))
    ap.add_argument("--adv", default="at", choices=ADV_METHODS)
    ap.add_argument("--reg", default="cosine", choices=REGULARIZERS + COMBO_REGULARIZERS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eval-only", action="store_true", help="Skip training")
    ap.add_argument("--run-all", action="store_true", help="Run all combinations")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: fewer eval steps/restarts/samples (~25 min for --run-all)")
    ap.add_argument("--fast-train", action="store_true",
                    help="Fast training: fewer PGD steps (3) and smaller step_size (0.01) for quicker runs")
    ap.add_argument("--lambda-igr", type=float, default=None,
                    help="Override lambda_igr (default: auto from paper per adv method)")
    ap.add_argument("--eval-adv", action="store_true",
                    help="Also evaluate adversarial accuracy (FGSM/PGD20/CW)")
    ap.add_argument("--eval-activation", action="store_true",
                    help="Also evaluate activation consistency")
    ap.add_argument("--run-combo", action="store_true",
                    help="Run combined regularizer experiments (cosine+diffkendall, cosine+soft_spearman)")
    ap.add_argument("--combo-w1", type=float, default=0.5,
                    help="Weight for first (vector) regularizer in combo (default: 0.5)")
    ap.add_argument("--combo-w2", type=float, default=0.5,
                    help="Weight for second (rank) regularizer in combo (default: 0.5)")
    ap.add_argument("--cross-proxy", action="store_true",
                    help="Cross-proxy IFIA evaluation: attack each model with cosine/kendall/spearman/pearson proxies")
    ap.add_argument("--cross-proxy-models", nargs="*", default=None,
                    help="Specific models to cross-proxy evaluate (default: all trained)")
    ap.add_argument("--export", default="results", help="Export directory")
    args = ap.parse_args()

    # --fast-train: maziau PGD zingsniu treniravime (greiciau, silpnesnis priesas)
    if args.fast_train:
        print(">>> FAST-TRAIN mode: PGD steps=3  step_size=0.01\n")

    # --quick perrasinea eval nustatymus DATASET_DEFAULTS'e
    if args.quick:
        for d in DATASET_DEFAULTS.values():
            d["eval_steps"] = 50
            d["eval_restarts"] = 1
            d["ig_steps_eval"] = 20
            d["eval_n_samples"] = 50
        print(">>> QUICK mode: eval_steps=50  restarts=1  n_samples=50  ig_eval=20\n")

    t_start = time.time()

    ran_something = False

    if args.run_all:
        combos = get_valid_combinations()
        all_results = {}
        all_adv_acc = {}
        all_activation = {}
        total = len(combos)
        for idx, (adv, reg) in enumerate(combos, 1):
            try:
                print(f"\n[{idx}/{total}]")
                label, metrics, extra = run_single(
                    args.dataset, adv, reg,
                    epochs=args.epochs, eval_only=args.eval_only, seed=args.seed,
                    lambda_igr=args.lambda_igr, fast_train=args.fast_train,
                    do_eval_adv=args.eval_adv, do_eval_activation=args.eval_activation,
                    quick=args.quick,
                    combo_w1=args.combo_w1, combo_w2=args.combo_w2,
                )
                all_results[label] = metrics
                if "adv_accuracy" in extra:
                    all_adv_acc[label] = extra["adv_accuracy"]
                if "activation" in extra:
                    all_activation[label] = {"overall": extra["activation"]["overall"]}
            except Exception as e:
                print(f"FAILED: {adv}+{reg}: {e}")
                import traceback; traceback.print_exc()
            finally:
                # GPU atminties atlaisvinimas tarp eksperimentu
                gc.collect()
                torch.cuda.empty_cache()

        # palyginimo lentele
        print(f"\n{'='*80}")
        print(f"  COMPARISON TABLE: {args.dataset}")
        if args.lambda_igr is not None:
            print(f"  lambda_igr: {args.lambda_igr} (manual override)")
        else:
            print(f"  lambda_igr: 1.0 (default for all methods)")
        print(f"{'='*80}")
        header = f"{'Model':<25} {'Top-k':>8} {'Kendall':>8} {'Spearman':>8} {'Cosine':>8} {'Pearson':>8}"
        print(header)
        print("-" * len(header))
        for label, m in all_results.items():
            print(f"{label:<25} {m['top_k']:>8.4f} {m['kendall_tau']:>8.4f} "
                  f"{m['spearman_rho']:>8.4f} {m['cosine_sim']:>8.4f} {m['pearson']:>8.4f}")

        if all_adv_acc:
            print(f"\n{'='*80}")
            print(f"  ADVERSARIAL ACCURACY: {args.dataset}")
            print(f"{'='*80}")
            header2 = f"{'Model':<25} {'Natural':>8} {'FGSM':>8} {'PGD-20':>8} {'CW∞':>8}"
            print(header2)
            print("-" * len(header2))
            for label, m in all_adv_acc.items():
                print(f"{label:<25} {m['natural']:>8.2%} {m['fgsm']:>8.2%} "
                      f"{m['pgd20']:>8.2%} {m['cw_inf']:>8.2%}")

        if all_activation:
            print(f"\n{'='*80}")
            print(f"  ACTIVATION CONSISTENCY: {args.dataset}")
            print(f"{'='*80}")
            for label, m in all_activation.items():
                print(f"  {label:<25} {m['overall']:.4f}")

        # eksportas
        export_dir = os.path.join(args.export, args.dataset)
        export_results(all_results, export_dir)
        # JSON dashboard'ui
        results_json_path = os.path.join(export_dir, "results.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results JSON saved: {results_json_path}")
        if all_adv_acc:
            import json as _json
            adv_path = os.path.join(export_dir, "adv_accuracy.json")
            with open(adv_path, "w", encoding="utf-8") as f:
                _json.dump(all_adv_acc, f, indent=2)
            print(f"Adv accuracy saved: {adv_path}")
        if all_activation:
            import json as _json
            act_path = os.path.join(export_dir, "activation_consistency.json")
            with open(act_path, "w", encoding="utf-8") as f:
                _json.dump(all_activation, f, indent=2)
            print(f"Activation consistency saved: {act_path}")
        ran_something = True

    if args.run_combo:
        # kombinuoti reg eksperimentai su AT/TRADES/MART
        combo_combos = []
        for adv in ["at", "trades", "mart"]:
            for combo_reg in COMBO_REGULARIZERS:
                combo_combos.append((adv, combo_reg))

        combo_results = {}
        combo_adv_acc = {}
        total = len(combo_combos)
        for idx, (adv, reg) in enumerate(combo_combos, 1):
            try:
                print(f"\n[{idx}/{total}] combo w1={args.combo_w1} w2={args.combo_w2}")
                label, metrics, extra = run_single(
                    args.dataset, adv, reg,
                    epochs=args.epochs, eval_only=args.eval_only, seed=args.seed,
                    lambda_igr=args.lambda_igr, fast_train=args.fast_train,
                    do_eval_adv=args.eval_adv, do_eval_activation=False,
                    quick=args.quick,
                    combo_w1=args.combo_w1, combo_w2=args.combo_w2,
                )
                combo_results[label] = metrics
                if "adv_accuracy" in extra:
                    combo_adv_acc[label] = extra["adv_accuracy"]
            except Exception as e:
                print(f"FAILED: {adv}+{reg}: {e}")
                import traceback; traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        print(f"\n{'='*80}")
        print(f"  COMBO RESULTS: {args.dataset} (w1={args.combo_w1}, w2={args.combo_w2})")
        print(f"{'='*80}")
        header = f"{'Model':<35} {'Top-k':>8} {'Kendall':>8} {'Spearman':>8} {'Cosine':>8} {'Pearson':>8}"
        print(header)
        print("-" * len(header))
        for label, m in combo_results.items():
            print(f"{label:<35} {m['top_k']:>8.4f} {m['kendall_tau']:>8.4f} "
                  f"{m['spearman_rho']:>8.4f} {m['cosine_sim']:>8.4f} {m['pearson']:>8.4f}")

        if combo_adv_acc:
            print(f"\n{'='*80}")
            print(f"  ADVERSARIAL ACCURACY (combo): {args.dataset}")
            print(f"{'='*80}")
            header2 = f"{'Model':<35} {'Natural':>8} {'FGSM':>8} {'PGD-20':>8} {'CW∞':>8}"
            print(header2)
            print("-" * len(header2))
            for label, m in combo_adv_acc.items():
                print(f"{label:<35} {m['natural']:>8.2%} {m['fgsm']:>8.2%} "
                      f"{m['pgd20']:>8.2%} {m['cw_inf']:>8.2%}")

        # combo eksportas
        export_dir = os.path.join(args.export, args.dataset)
        os.makedirs(export_dir, exist_ok=True)
        combo_path = os.path.join(export_dir, "combo_results.json")
        with open(combo_path, "w", encoding="utf-8") as f:
            json.dump(combo_results, f, indent=2)
        print(f"Combo results saved: {combo_path}")
        if combo_adv_acc:
            adv_path = os.path.join(export_dir, "combo_adv_accuracy.json")
            with open(adv_path, "w", encoding="utf-8") as f:
                json.dump(combo_adv_acc, f, indent=2)
        ran_something = True

    if args.cross_proxy:
        # Cross-proxy IFIA: kiekviena modeli atakuojam keliais proxy
        print(f"\n{'='*80}")
        print(f"  CROSS-PROXY IFIA EVALUATION: {args.dataset}")
        print(f"  Proxies: {CROSS_PROXIES}")
        print(f"{'='*80}\n")

        # pasirenkam, kuriuos modelius vertinti
        if args.cross_proxy_models:
            model_specs = []
            for m in args.cross_proxy_models:
                if "+" in m:
                    parts = m.split("+", 1)
                    model_specs.append((parts[0], parts[1]))
                else:
                    model_specs.append((m, "none"))
        else:
            # visos galiojancios kombinacijos
            model_specs = get_valid_combinations()
            # ir kombinuoti reg
            for adv in ["at", "trades", "mart"]:
                for combo_reg in COMBO_REGULARIZERS:
                    model_specs.append((adv, combo_reg))

        cross_matrix = {}  # {model_label: {proxy: {metric: float}}}
        total = len(model_specs)
        for idx, (adv, reg) in enumerate(model_specs, 1):
            label = f"{adv}+{reg}" if reg != "none" else adv
            reg_safe = reg.replace("+", "_") if reg else reg
            ckpt_path = f"artifacts/{args.dataset}_{adv}_{reg_safe}.pt"
            if not os.path.exists(ckpt_path):
                print(f"  [{idx}/{total}] SKIP {label} (no checkpoint)")
                continue

            print(f"\n[{idx}/{total}] Cross-proxy: {label}")
            cfg = make_config(args.dataset, adv, reg, epochs=args.epochs, seed=args.seed,
                              lambda_igr=args.lambda_igr)
            cfg_path = save_temp_config(cfg)
            try:
                proxy_results = cross_proxy_evaluate(cfg_path, proxies=CROSS_PROXIES)
                cross_matrix[label] = proxy_results
            except Exception as e:
                print(f"  FAILED: {label}: {e}")
                import traceback; traceback.print_exc()
            finally:
                os.unlink(cfg_path)
                gc.collect()
                torch.cuda.empty_cache()

        # cross-proxy matrica
        if cross_matrix:
            metrics_to_show = ["top_k", "kendall_tau", "spearman_rho", "cosine_sim", "pearson"]
            for metric in metrics_to_show:
                print(f"\n{'='*80}")
                print(f"  CROSS-PROXY MATRIX: {metric} ({args.dataset})")
                print(f"  Rows=models, Cols=attack proxy")
                print(f"{'='*80}")
                header = f"{'Model':<30} " + " ".join(f"{p:>14}" for p in CROSS_PROXIES)
                print(header)
                print("-" * len(header))
                for label, proxy_res in cross_matrix.items():
                    vals = " ".join(f"{proxy_res[p][metric]:>14.4f}" for p in CROSS_PROXIES)
                    print(f"{label:<30} {vals}")

            # issaugom
            export_dir = os.path.join(args.export, args.dataset)
            os.makedirs(export_dir, exist_ok=True)
            cp_path = os.path.join(export_dir, "cross_proxy_results.json")
            with open(cp_path, "w", encoding="utf-8") as f:
                json.dump(cross_matrix, f, indent=2)
            print(f"\nCross-proxy results saved: {cp_path}")
        ran_something = True

    if not ran_something:
        label, metrics, extra = run_single(
            args.dataset, args.adv, args.reg,
            epochs=args.epochs, eval_only=args.eval_only, seed=args.seed,
            lambda_igr=args.lambda_igr, fast_train=args.fast_train,
            do_eval_adv=args.eval_adv, do_eval_activation=args.eval_activation,
            quick=args.quick,
            combo_w1=args.combo_w1, combo_w2=args.combo_w2,
        )
        export_dir = os.path.join(args.export, args.dataset)
        export_results({label: metrics}, export_dir)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
