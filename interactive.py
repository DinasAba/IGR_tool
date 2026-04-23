#!/usr/bin/env python
# Interaktyvi CLI sasaja konkordacijos matu eksperimentams.
# Paleidimas: python interactive.py

import gc
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (
    DATASET_DEFAULTS,
    LAMBDA_IGR_DEFAULTS,
    ADV_METHODS,
    REGULARIZERS,
    make_config,
    save_temp_config,
    get_valid_combinations,
)
from src.train import main as train_main
from src.eval_ifia import evaluate, export_results


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str):
    w = 60
    print()
    print("=" * w)
    print(f"  {title}".center(w))
    print("=" * w)


def pick_one(prompt: str, options: list[str], allow_back: bool = True) -> str | None:
    # numeruotas meniu; grazina pasirinkima arba None jei 'back'
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    if allow_back:
        print(f"  [0] Back")
    while True:
        try:
            choice = input("\n> ").strip()
            if choice == "0" and allow_back:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, EOFError):
            pass
        print("  Invalid choice. Try again.")


def pick_many(prompt: str, options: list[str]) -> list[str]:
    # keli pasirinkimai, atskirti kableliais
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print(f"  [a] All")
    print(f"  [0] Back")
    while True:
        try:
            choice = input("\n> ").strip().lower()
            if choice == "0":
                return []
            if choice == "a":
                return list(options)
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = [options[i] for i in indices if 0 <= i < len(options)]
            if selected:
                return selected
        except (ValueError, EOFError):
            pass
        print("  Enter numbers separated by commas (e.g. 1,3,4) or 'a' for all.")


def input_int(prompt: str, default: int) -> int:
    while True:
        try:
            val = input(f"{prompt} [{default}]: ").strip()
            return int(val) if val else default
        except (ValueError, EOFError):
            print("  Enter a valid integer.")


def input_float(prompt: str, default: float) -> float:
    while True:
        try:
            val = input(f"{prompt} [{default}]: ").strip()
            return float(val) if val else default
        except (ValueError, EOFError):
            print("  Enter a valid number.")


def yn(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    val = input(f"{prompt} [{d}]: ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


def page_train_single(dataset: str):
    # vieno modelio treniravimas
    print_header(f"Train Single Model ({dataset})")

    adv = pick_one("Adversarial method:", ADV_METHODS, allow_back=True)
    if adv is None:
        return

    if adv == "none":
        reg = "none"
    else:
        reg = pick_one("Regularizer:", REGULARIZERS, allow_back=True)
        if reg is None:
            return

    epochs = input_int("Epochs", DATASET_DEFAULTS[dataset]["epochs"])
    lam = input_float("Lambda IGR", LAMBDA_IGR_DEFAULTS.get(adv, 1.0))

    label = f"{adv}+{reg}" if reg != "none" else adv
    print(f"\n  Will train: {label}  epochs={epochs}  lambda={lam}")
    if not yn("Proceed?"):
        return

    cfg = make_config(dataset, adv, reg, epochs=epochs, lambda_igr=lam)
    cfg["data"]["num_workers"] = 0
    cfg_path = save_temp_config(cfg)

    try:
        t0 = time.time()
        train_main(cfg_path)
        print(f"\n  Training complete in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\n  Training FAILED: {e}")
    finally:
        os.unlink(cfg_path)
        gc.collect()
        torch.cuda.empty_cache()

    input("\nPress Enter to continue...")


def page_train_all(dataset: str):
    # visu 16 kombinaciju treniravimas
    print_header(f"Train All Combinations ({dataset})")

    combos = get_valid_combinations()
    epochs = input_int("Epochs", DATASET_DEFAULTS[dataset]["epochs"])
    fast = yn("Fast training (PGD steps=3)?", default=False)

    print(f"\n  Will train {len(combos)} models, {epochs} epochs each.")
    if fast:
        print("  Fast mode: PGD steps=3, step_size=0.01")
    if not yn("Proceed?"):
        return

    t0 = time.time()
    for idx, (adv, reg) in enumerate(combos, 1):
        label = f"{adv}+{reg}" if reg != "none" else adv
        print(f"\n[{idx}/{len(combos)}] Training: {label}")

        cfg = make_config(dataset, adv, reg, epochs=epochs, fast_train=fast)
        cfg["data"]["num_workers"] = 0
        cfg_path = save_temp_config(cfg)

        try:
            train_main(cfg_path)
        except Exception as e:
            print(f"  FAILED: {e}")
        finally:
            os.unlink(cfg_path)
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n  All training done in {(time.time()-t0)/60:.1f} min")
    input("\nPress Enter to continue...")


def page_evaluate(dataset: str):
    # IFIA vertinimas treniruotiems modeliams
    print_header(f"Evaluate Models ({dataset})")

    # ieskom treniruotu modeliu
    artifacts = "artifacts"
    available = []
    if os.path.isdir(artifacts):
        for f in sorted(os.listdir(artifacts)):
            if f.startswith(dataset + "_") and f.endswith(".pt"):
                name = f[len(dataset)+1:-3]
                parts = name.split("_", 1)
                if len(parts) == 2:
                    adv, reg = parts
                    label = f"{adv}+{reg}" if reg != "none" else adv
                else:
                    label = parts[0]
                available.append((label, adv, reg if len(parts) == 2 else "none"))

    if not available:
        print("  No trained models found. Train some models first.")
        input("\nPress Enter to continue...")
        return

    print(f"  Found {len(available)} trained model(s).")
    labels = [a[0] for a in available]
    selected_labels = pick_many("Select models to evaluate:", labels)
    if not selected_labels:
        return

    quick = yn("Quick eval (fewer samples, 1 restart)?", default=True)

    # quick rezime - maziau sample'u, 1 restart, trumpesnis IG
    if quick:
        for d in DATASET_DEFAULTS.values():
            d["eval_steps"] = 50
            d["eval_restarts"] = 1
            d["ig_steps_eval"] = 20
            d["eval_n_samples"] = 50

    all_results = {}
    total = len(selected_labels)

    t0 = time.time()
    for idx, label in enumerate(selected_labels, 1):
        match = next((a for a in available if a[0] == label), None)
        if not match:
            continue
        _, adv, reg = match
        print(f"\n[{idx}/{total}] Evaluating: {label}")

        cfg = make_config(dataset, adv, reg, epochs=1)
        cfg["data"]["num_workers"] = 0
        cfg_path = save_temp_config(cfg)

        try:
            res = evaluate(cfg_path)
            all_results.update(res)
        except Exception as e:
            print(f"  FAILED: {e}")
        finally:
            os.unlink(cfg_path)
            gc.collect()
            torch.cuda.empty_cache()

    if all_results:
        print(f"\n{'='*80}")
        print(f"  RESULTS: {dataset}  (IFIA evaluation)")
        print(f"{'='*80}")
        header = f"{'Model':<25} {'Top-k':>8} {'Kendall':>8} {'Spearman':>8} {'Cosine':>8} {'Pearson':>8}"
        print(header)
        print("-" * len(header))
        for label, m in all_results.items():
            print(f"{label:<25} {m['top_k']:>8.4f} {m['kendall_tau']:>8.4f} "
                  f"{m['spearman_rho']:>8.4f} {m['cosine_sim']:>8.4f} {m['pearson']:>8.4f}")

        export_dir = os.path.join("results", dataset)
        export_results(all_results, export_dir)
        print(f"\n  Saved to: {export_dir}/")

    print(f"\n  Evaluation done in {(time.time()-t0)/60:.1f} min")
    input("\nPress Enter to continue...")


def page_view_results(dataset: str):
    # perzvalga esamiems rezultatams
    print_header(f"Results: {dataset}")

    results_path = os.path.join("results", dataset, "results.json")
    if not os.path.exists(results_path):
        print("  No results found. Run evaluation first.")
        input("\nPress Enter to continue...")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("  Results file is empty. Run evaluation first.")
        input("\nPress Enter to continue...")
        return

    # Straipsnio metrikos (Table 2)
    print("\n  Paper metrics (Table 2: top-k + Kendall tau):")
    print(f"  {'Model':<25} {'Top-k':>8} {'Kendall':>8}")
    print("  " + "-" * 43)
    for label, m in data.items():
        print(f"  {label:<25} {m['top_k']:>8.4f} {m['kendall_tau']:>8.4f}")

    # Papildomos metrikos
    print(f"\n  Extension metrics (Spearman, Cosine, Pearson):")
    print(f"  {'Model':<25} {'Spearman':>8} {'Cosine':>8} {'Pearson':>8}")
    print("  " + "-" * 51)
    for label, m in data.items():
        print(f"  {label:<25} {m.get('spearman_rho', 0):>8.4f} "
              f"{m.get('cosine_sim', 0):>8.4f} {m.get('pearson', 0):>8.4f}")

    # Adversarinis tikslumas (6.3 skyrius)
    adv_path = os.path.join("results", dataset, "adv_accuracy.json")
    if os.path.exists(adv_path):
        with open(adv_path, "r", encoding="utf-8") as f:
            adv_data = json.load(f)
        if adv_data:
            print(f"\n  Adversarial accuracy (Section 6.3):")
            print(f"  {'Model':<25} {'Natural':>8} {'FGSM':>8} {'PGD-20':>8} {'CW-inf':>8}")
            print("  " + "-" * 59)
            for label, m in adv_data.items():
                print(f"  {label:<25} {m['natural']:>8.2%} {m['fgsm']:>8.2%} "
                      f"{m['pgd20']:>8.2%} {m['cw_inf']:>8.2%}")

    input("\nPress Enter to continue...")


def page_dataset_menu(dataset: str):
    # submeniu vienam duomenu rinkiniui
    while True:
        clear_screen()
        print_header(f"Dataset: {dataset.upper()}")

        # busena
        n_models = 0
        if os.path.isdir("artifacts"):
            n_models = sum(1 for f in os.listdir("artifacts")
                         if f.startswith(dataset + "_") and f.endswith(".pt"))
        has_results = os.path.exists(os.path.join("results", dataset, "results.json"))

        print(f"  Trained models: {n_models}")
        print(f"  Results: {'Available' if has_results else 'Not yet'}")

        action = pick_one("What do you want to do?", [
            "Train single model",
            "Train all 16 combinations",
            "Evaluate models (IFIA)",
            "View results",
        ], allow_back=True)

        if action is None:
            return
        elif action == "Train single model":
            page_train_single(dataset)
        elif action == "Train all 16 combinations":
            page_train_all(dataset)
        elif action == "Evaluate models (IFIA)":
            page_evaluate(dataset)
        elif action == "View results":
            page_view_results(dataset)


def main():
    while True:
        clear_screen()
        print_header("Concordance Measure Comparison Tool")
        print()
        print("  Based on: Wang & Kong, NeurIPS 2022")
        print("  Metrics: Top-k intersection, Kendall tau, Spearman rho,")
        print("           Cosine similarity, Pearson correlation")
        print()

        datasets = list(DATASET_DEFAULTS.keys())
        dataset = pick_one("Select dataset:", datasets, allow_back=False)

        if dataset is None:
            break

        page_dataset_menu(dataset)

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
