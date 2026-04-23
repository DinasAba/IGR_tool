# Streamlit dashboard'as: python -m streamlit run dashboard.py

import gc
import json
import os
import sys
import tempfile

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (
    DATASET_DEFAULTS, LAMBDA_IGR_DEFAULTS, make_config, save_temp_config,
    get_valid_combinations, COMBO_REGULARIZERS,
)

st.set_page_config(
    page_title="Concordance Measure Comparison",
    page_icon="",
    layout="wide",
)

ADV_METHODS = ["none", "at", "trades", "mart"]
REGULARIZERS = ["none", "cosine", "diff_kendall", "soft_spearman", "pearson"] + COMBO_REGULARIZERS
RESULTS_DIR = "results"
ARTIFACTS_DIR = "artifacts"

METRIC_NAMES = {
    "top_k": "Top-k Intersection",
    "kendall_tau": "Kendall τ",
    "spearman_rho": "Spearman ρ",
    "cosine_sim": "Cosine Similarity",
    "pearson": "Pearson r",
}

ADV_ACC_NAMES = {
    "natural": "Natural",
    "fgsm": "FGSM",
    "pgd20": "PGD-20",
    "cw_inf": "CW∞",
}


def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if data else None  # {} laikom kaip None
    return None


def load_results(dataset: str) -> dict | None:
    base = load_json(os.path.join(RESULTS_DIR, dataset, "results.json"))
    combo = load_json(os.path.join(RESULTS_DIR, dataset, "combo_results.json"))
    if base and combo:
        base.update(combo)
        return base
    return base or combo

def load_adv_results(dataset: str) -> dict | None:
    return load_json(os.path.join(RESULTS_DIR, dataset, "adv_accuracy.json"))

def load_activation_results(dataset: str) -> dict | None:
    return load_json(os.path.join(RESULTS_DIR, dataset, "activation_consistency.json"))

def load_cross_proxy_results(dataset: str) -> dict | None:
    return load_json(os.path.join(RESULTS_DIR, dataset, "cross_proxy_results.json"))


VALID_ADV_METHODS = {"none", "at", "trades", "mart"}

def get_available_models(dataset: str) -> list[str]:
    models = []
    if not os.path.isdir(ARTIFACTS_DIR):
        return models
    for f in os.listdir(ARTIFACTS_DIR):
        if f.startswith(dataset + "_") and f.endswith(".pt"):
            name = f[len(dataset)+1:-3]
            parts = name.split("_", 1)
            if len(parts) == 2:
                adv, reg = parts
                if adv not in VALID_ADV_METHODS:
                    continue
                # failo vardas naudoja "_" vietoj "+" kombinuotiems reguliarizatoriams
                for combo in COMBO_REGULARIZERS:
                    combo_safe = combo.replace("+", "_")
                    if reg == combo_safe:
                        reg = combo
                        break
                label = f"{adv}+{reg}" if reg != "none" else adv
            else:
                if parts[0] not in VALID_ADV_METHODS:
                    continue
                label = parts[0]
            models.append(label)
    return sorted(models)


def _parse_model_label(label: str) -> tuple[str, str]:
    # 'at+cosine+diff_kendall' -> ('at', 'cosine+diff_kendall')
    for combo in COMBO_REGULARIZERS:
        if label.endswith("+" + combo):
            adv = label[: -(len(combo) + 1)]
            return adv, combo
    if "+" in label:
        adv, reg = label.split("+", 1)
        return adv, reg
    return label, "none"


def _model_sort_key(model_name: str) -> tuple:
    ADV_ORDER = {"none": 0, "at": 1, "trades": 2, "mart": 3}
    REG_ORDER = {
        "none": 0, "cosine": 1, "diff_kendall": 2, "soft_spearman": 3, "pearson": 4,
        "cosine+diff_kendall": 5, "cosine+soft_spearman": 6,
    }
    for combo in COMBO_REGULARIZERS:
        if model_name.endswith("+" + combo):
            adv = model_name[: -(len(combo) + 1)]
            return (ADV_ORDER.get(adv, 9), REG_ORDER.get(combo, 9))
    if "+" in model_name:
        adv, reg = model_name.split("+", 1)
        return (ADV_ORDER.get(adv, 9), REG_ORDER.get(reg, 9))
    return (ADV_ORDER.get(model_name, 9), 0)


def results_to_df(results: dict) -> pd.DataFrame:
    rows = [{"Model": m, **v} for m, v in results.items()]
    df = pd.DataFrame(rows)
    df["_sort"] = df["Model"].apply(_model_sort_key)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def _get_adv_group(model_name: str) -> str:
    adv, _ = _parse_model_label(model_name)
    return adv


def _split_by_group(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df["_group"] = df["Model"].apply(_get_adv_group)
    groups = {}
    for group_name in ["none", "at", "trades", "mart"]:
        sub = df[df["_group"] == group_name].drop(columns=["_group"]).copy()
        if not sub.empty:
            groups[group_name] = sub.reset_index(drop=True)
    df.drop(columns=["_group"], inplace=True)
    return groups


GROUP_DISPLAY = {"none": "Standard", "at": "AT", "trades": "TRADES", "mart": "MART"}


st.sidebar.title("Settings")
dataset = st.sidebar.selectbox("Dataset", list(DATASET_DEFAULTS.keys()))
available_models = get_available_models(dataset)

st.sidebar.markdown("---")
st.sidebar.subheader("Train New Model")

col_adv, col_reg = st.sidebar.columns(2)
with col_adv:
    adv_method = st.selectbox("Adv Method", ["All"] + ADV_METHODS)
with col_reg:
    regularizer = st.selectbox("Regularizer", ["All"] + REGULARIZERS)

epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=200, value=10)
lambda_igr = st.sidebar.number_input("lambda (IGR)", min_value=0.0, max_value=10.0,
                                      value=LAMBDA_IGR_DEFAULTS.get(adv_method, 1.0), step=0.1)

if st.sidebar.button("Train", type="primary"):
    from src.train import main as train_main

    # (adv, reg) kombinacijos treniravimui
    if adv_method == "All" or regularizer == "All":
        if adv_method == "All" and regularizer == "All":
            combos = get_valid_combinations()
            for adv in ADV_METHODS:
                if adv != "none":
                    for cr in COMBO_REGULARIZERS:
                        combos.append((adv, cr))
        elif adv_method == "All":
            combos = [(a, regularizer) for a in ADV_METHODS]
        else:
            combos = [(adv_method, r) for r in REGULARIZERS]
    else:
        combos = [(adv_method, regularizer)]

    total = len(combos)
    progress = st.sidebar.progress(0)
    trained, failed = 0, 0

    for idx, (adv, reg) in enumerate(combos):
        label = f"{adv}+{reg}" if reg != "none" else adv
        progress.progress(idx / total, text=f"Training {label} ({idx+1}/{total})...")

        cfg = make_config(dataset, adv, reg, epochs=epochs, lambda_igr=lambda_igr)
        cfg["data"]["num_workers"] = 0
        cfg_path = save_temp_config(cfg)

        try:
            train_main(cfg_path)
            trained += 1
        except Exception as e:
            st.sidebar.warning(f"Failed {label}: {e}")
            failed += 1
        finally:
            os.unlink(cfg_path)
            gc.collect()
            torch.cuda.empty_cache()

    progress.empty()
    msg = f"Done: {trained} trained"
    if failed:
        msg += f", {failed} failed"
    st.sidebar.success(msg)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Evaluate")

EVAL_TYPES_INFO = {
    "Attribution Robustness (IFIA)": "Single-proxy IFIA attack (cosine). Measures how well explanations survive adversarial perturbation.",
    "Cross-Proxy IFIA": "Attack each model with 4 proxy types (cosine/kendall/spearman/pearson). Shows measure-dependence.",
    "Adversarial Accuracy (FGSM/PGD/CW)": "Classification accuracy under FGSM, PGD-20, CW∞ attacks.",
}
EVAL_TYPES = list(EVAL_TYPES_INFO.keys())

eval_type = st.sidebar.selectbox("Evaluation Type", ["All"] + EVAL_TYPES)

if eval_type != "All" and eval_type in EVAL_TYPES_INFO:
    st.sidebar.caption(EVAL_TYPES_INFO[eval_type])

eval_quick = st.sidebar.checkbox("Quick eval (fewer samples)", value=True,
                                  help="50 samples, 50 steps, 1 restart (vs 200/200/5 full)")

select_all_models = st.sidebar.checkbox("Select all models", value=False)
if select_all_models:
    eval_models = available_models
    st.sidebar.caption(f"✓ {len(available_models)} models selected")
else:
    eval_models = st.sidebar.multiselect("Models to evaluate", available_models,
                                         default=available_models[:4] if available_models else [])

# laiko prognoze (grubus ivertinimas pagal eksperimento patirti)
if eval_models and eval_type:
    n_models = len(eval_models)
    n_evals = len(EVAL_TYPES) if eval_type == "All" else 1
    if eval_quick:
        est_per_model = {"Attribution Robustness (IFIA)": 1.5,
                         "Cross-Proxy IFIA": 6,
                         "Adversarial Accuracy (FGSM/PGD/CW)": 2}
    else:
        est_per_model = {"Attribution Robustness (IFIA)": 7,
                         "Cross-Proxy IFIA": 28,
                         "Adversarial Accuracy (FGSM/PGD/CW)": 4}
    if eval_type == "All":
        total_min = sum(est_per_model.values()) * n_models
    else:
        total_min = est_per_model.get(eval_type, 5) * n_models
    st.sidebar.caption(f"⏱ ~{total_min:.0f} min estimated")

if st.sidebar.button("Run Evaluation"):
    if not eval_models:
        st.sidebar.warning("Select at least one model")
    else:
        eval_types_to_run = EVAL_TYPES if eval_type == "All" else [eval_type]

        all_ifia_results = {}
        all_adv_results = {}
        all_cp_results = {}
        all_act_results = {}

        total_steps = len(eval_models) * len(eval_types_to_run)
        progress = st.progress(0)
        step = 0

        for current_eval in eval_types_to_run:
            for i, model_label in enumerate(eval_models):
                progress.progress(
                    step / total_steps,
                    text=f"{current_eval[:20]}... {model_label} ({step+1}/{total_steps})",
                )

                adv, reg = _parse_model_label(model_label)

                cfg = make_config(dataset, adv, reg, epochs=1)
                cfg["data"]["num_workers"] = 0
                if eval_quick:
                    cfg["eval"]["n_samples"] = 50
                    cfg["eval"]["attack"]["steps"] = 50
                    cfg["eval"]["attack"]["restarts"] = 1

                cfg_path = save_temp_config(cfg)

                try:
                    if current_eval == "Attribution Robustness (IFIA)":
                        from src.eval_ifia import evaluate
                        res = evaluate(cfg_path)
                        all_ifia_results.update(res)
                    elif current_eval == "Adversarial Accuracy (FGSM/PGD/CW)":
                        from src.eval_adversarial import evaluate_adv
                        batch_lim = 10 if eval_quick else None
                        res = evaluate_adv(cfg_path, batch_limit=batch_lim)
                        all_adv_results.update(res)
                    elif current_eval == "Cross-Proxy IFIA":
                        from src.eval_ifia import cross_proxy_evaluate
                        res = cross_proxy_evaluate(cfg_path)
                        all_cp_results[model_label] = res
                    elif current_eval == "Activation Consistency":
                        from src.eval_activation import evaluate_activation
                        n = 50 if eval_quick else 200
                        res = evaluate_activation(cfg_path, n_samples=n)
                        all_act_results.update(res)
                except Exception as e:
                    st.warning(f"Failed {model_label} ({current_eval}): {e}")
                finally:
                    os.unlink(cfg_path)
                    gc.collect()
                    torch.cuda.empty_cache()

                step += 1

        progress.progress(1.0)

        out_dir = os.path.join(RESULTS_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)

        if all_ifia_results:
            path = os.path.join(out_dir, "results.json")
            existing = load_results(dataset) or {}
            existing.update(all_ifia_results)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)

        if all_adv_results:
            path = os.path.join(out_dir, "adv_accuracy.json")
            existing = load_adv_results(dataset) or {}
            existing.update(all_adv_results)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)

        if all_cp_results:
            path = os.path.join(out_dir, "cross_proxy_results.json")
            existing = load_cross_proxy_results(dataset) or {}
            existing.update(all_cp_results)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)

        if all_act_results:
            flat = {}
            for lbl, data in all_act_results.items():
                flat[lbl] = {"overall": data["overall"]}
            path = os.path.join(out_dir, "activation_consistency.json")
            existing = load_activation_results(dataset) or {}
            existing.update(flat)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)

        progress.empty()
        st.sidebar.success("Evaluation complete!")
        st.rerun()


st.title("Concordance Measure Comparison Tool")
st.markdown(f"**Dataset:** `{dataset}` | **Trained models:** {len(available_models)}")

tab1, tab2, tab_cp, tab_vis, tab4 = st.tabs([
    "Attribution Robustness",
    "Adversarial Accuracy",
    "Cross-Proxy IFIA",
    "IG Visualization",
    "About",
])


with tab1:
    attr_data = load_results(dataset)

    if attr_data is None:
        st.info("No attribution robustness results yet. Run evaluation from the sidebar.")
    else:
        df = results_to_df(attr_data)
        groups = _split_by_group(df)

        st.subheader("Attribution Robustness (IFIA)")
        st.caption("Best value per group highlighted in green. Higher = better.")

        for group_key, group_df in groups.items():
            st.markdown(f"#### {GROUP_DISPLAY[group_key]}")
            gdf = group_df.set_index("Model").rename(columns=METRIC_NAMES)
            styled = gdf.style.format("{:.4f}")
            if len(gdf) > 1:
                styled = styled.highlight_max(axis=0, color="#c6efce")
            st.dataframe(styled, use_container_width=True,
                         height=(len(gdf) + 1) * 38 + 10)

        st.subheader("Heatmap: Models vs Metrics")
        df_all = df.set_index("Model").rename(columns=METRIC_NAMES)
        fig_heat = px.imshow(
            df_all.values.astype(float),
            labels=dict(x="Metric", y="Model", color="Value"),
            x=list(df_all.columns),
            y=list(df_all.index),
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            aspect="auto",
            text_auto=".3f",
        )
        fig_heat.update_layout(height=max(300, len(df_all) * 35 + 100))
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("LaTeX Table"):
            latex = df_all.to_latex(float_format="%.4f", bold_rows=True)
            st.code(latex, language="latex")


with tab2:
    adv_data = load_adv_results(dataset)

    if adv_data is None:
        st.info("No adversarial accuracy results yet. Select 'Adversarial Accuracy' in sidebar and run evaluation.")
    else:
        df_adv = results_to_df(adv_data)
        adv_cols = [c for c in df_adv.columns if c != "Model"]
        adv_groups = _split_by_group(df_adv)

        st.subheader("Adversarial Accuracy")
        st.caption("Best value per group highlighted in green. Higher = better.")

        for group_key, group_df in adv_groups.items():
            st.markdown(f"#### {GROUP_DISPLAY[group_key]}")
            gdf = group_df.set_index("Model").rename(columns=ADV_ACC_NAMES)
            styled = gdf.style.format("{:.2%}")
            if len(gdf) > 1:
                styled = styled.highlight_max(axis=0, color="#c6efce")
            st.dataframe(styled, use_container_width=True,
                         height=(len(gdf) + 1) * 38 + 10)

        df_adv_all = df_adv.set_index("Model").rename(columns=ADV_ACC_NAMES)
        st.subheader("Accuracy Heatmap")
        fig_adv_heat = px.imshow(
            df_adv_all.values.astype(float),
            labels=dict(x="Attack", y="Model", color="Accuracy"),
            x=list(df_adv_all.columns),
            y=list(df_adv_all.index),
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            aspect="auto",
            text_auto=".1%",
        )
        fig_adv_heat.update_layout(height=max(300, len(df_adv_all) * 35 + 100))
        st.plotly_chart(fig_adv_heat, use_container_width=True)

        # lollipop grafikas: natural vs PGD-20
        if "natural" in adv_cols:
            st.subheader("Robustness Gap (Natural - PGD-20)")
            df_gap = df_adv.copy()
            attack_col = "pgd20" if "pgd20" in adv_cols else adv_cols[-1]
            df_gap["gap"] = df_gap["natural"] - df_gap[attack_col]
            df_gap = df_gap.sort_values("gap")

            fig_gap = go.Figure()
            for _, row in df_gap.iterrows():
                fig_gap.add_trace(go.Scatter(
                    x=[row["natural"], row[attack_col]],
                    y=[row["Model"], row["Model"]],
                    mode="lines+markers",
                    marker=dict(size=10),
                    line=dict(width=3),
                    name=row["Model"],
                    showlegend=False,
                    hovertemplate=(
                        f"Natural: {row['natural']:.1%}<br>"
                        f"PGD-20: {row[attack_col]:.1%}<br>"
                        f"Gap: {row['gap']:.1%}"
                    ),
                ))
            fig_gap.update_layout(
                xaxis_title="Accuracy",
                xaxis_tickformat=".0%",
                xaxis_range=[0, 1.05],
                height=max(300, len(df_gap) * 30 + 100),
                title="Natural vs PGD-20 Accuracy (gap = robustness cost)",
            )
            st.plotly_chart(fig_gap, use_container_width=True)


with tab_cp:
    cp_data = load_cross_proxy_results(dataset)

    if cp_data is None:
        st.info(
            "No cross-proxy results yet. Run from CLI:\n\n"
            "`python run_experiment.py --dataset mnist --cross-proxy`"
        )
    else:
        st.subheader("Cross-Proxy IFIA Evaluation")
        st.markdown(
            "Each matrix shows how well each model preserves explanation similarity "
            "when attacked by different IFIA proxy types. "
            "**Rows** = models (trained defense), **Columns** = attack proxy used."
        )

        # cp_data[model][proxy] = {metric: value}
        models_cp = sorted(cp_data.keys(), key=_model_sort_key)
        first_model = next(iter(cp_data.values()))
        proxies = list(first_model.keys())
        first_proxy_data = next(iter(first_model.values()))
        metrics_cp = list(first_proxy_data.keys())

        selected_metric = st.selectbox(
            "Select metric to display",
            metrics_cp,
            format_func=lambda m: METRIC_NAMES.get(m, m),
        )

        matrix_data = []
        for model in models_cp:
            row = {}
            for proxy in proxies:
                val = cp_data.get(model, {}).get(proxy, {}).get(selected_metric, float("nan"))
                row[proxy] = val
            matrix_data.append(row)

        df_cp = pd.DataFrame(matrix_data, index=models_cp, columns=proxies)

        st.dataframe(
            df_cp.style.format("{:.4f}")
            .highlight_max(axis=0, color="#c6efce")
            .highlight_min(axis=0, color="#ffc7ce"),
            use_container_width=True,
            height=(len(df_cp) + 1) * 38 + 10,
        )

        proxy_labels = [p.replace("_", " ").title() for p in proxies]
        fig_cp = px.imshow(
            df_cp.values.astype(float),
            labels=dict(x="Attack Proxy", y="Model (Defense)", color=METRIC_NAMES.get(selected_metric, selected_metric)),
            x=proxy_labels,
            y=models_cp,
            color_continuous_scale="RdYlGn",
            zmin=0,
            zmax=1,
            aspect="auto",
            text_auto=".3f",
        )
        fig_cp.update_layout(
            height=max(350, len(models_cp) * 50 + 120),
            title=f"Cross-Proxy Matrix: {METRIC_NAMES.get(selected_metric, selected_metric)}",
        )
        st.plotly_chart(fig_cp, use_container_width=True)

        with st.expander("All Metrics (compact tables)"):
            cols = st.columns(min(len(metrics_cp), 3))
            for i, metric in enumerate(metrics_cp):
                col = cols[i % len(cols)]
                mat = []
                for model in models_cp:
                    row = {}
                    for proxy in proxies:
                        row[proxy] = cp_data.get(model, {}).get(proxy, {}).get(metric, float("nan"))
                    mat.append(row)
                df_m = pd.DataFrame(mat, index=models_cp, columns=proxies)
                col.markdown(f"**{METRIC_NAMES.get(metric, metric)}**")
                col.dataframe(df_m.style.format("{:.3f}"), height=250)

        with st.expander("LaTeX Table"):
            latex_cp = df_cp.to_latex(float_format="%.4f", bold_rows=True)
            st.code(latex_cp, language="latex")


with tab_vis:
    st.subheader("Attribution Robustness Visualization")
    st.markdown(
        "Compare IG attributions on **clean** vs **adversarial** images for a baseline model "
        "(without IGR) and an IGR-regularized model. Similar to Figure 3 in Wang & Kong (2022)."
    )

    if not available_models:
        st.info("No trained models found. Train models first.")
    else:
        # bazinis = be reguliarizatoriaus, IGR = su reguliarizatoriumi
        baseline_models = [m for m in available_models if "+" not in m or m.count("+") == 0]
        if not baseline_models:
            baseline_models = available_models

        igr_models = [m for m in available_models if "+" in m]
        if not igr_models:
            igr_models = available_models

        col1, col2 = st.columns(2)
        with col1:
            vis_baseline = st.selectbox(
                "Baseline model (no regularizer)",
                baseline_models,
                index=min(1, len(baseline_models) - 1) if len(baseline_models) > 1 else 0,
                key="vis_baseline",
            )
        with col2:
            vis_igr = st.selectbox(
                "IGR model (with regularizer)",
                igr_models,
                index=0,
                key="vis_igr",
            )

        col3, col4, col5 = st.columns(3)
        with col3:
            vis_n_samples = st.number_input("Number of examples", min_value=1, max_value=10, value=3, key="vis_n")
        with col4:
            vis_ig_steps = st.number_input("IG steps", min_value=10, max_value=200, value=50, key="vis_ig")
        with col5:
            vis_pgd_steps = st.number_input("PGD steps", min_value=5, max_value=100, value=20, key="vis_pgd")

        vis_seed = st.number_input("Random seed (change for different samples)", min_value=0, value=42, key="vis_seed")

        if st.button("Generate Visualization", type="primary", key="vis_generate"):
            from src.visualize_ig import generate_ig_visualization

            adv_b, reg_b = _parse_model_label(vis_baseline)
            adv_i, reg_i = _parse_model_label(vis_igr)
            reg_b_safe = reg_b.replace("+", "_")
            reg_i_safe = reg_i.replace("+", "_")
            ckpt_base = os.path.join(ARTIFACTS_DIR, f"{dataset}_{adv_b}_{reg_b_safe}.pt")
            ckpt_igr = os.path.join(ARTIFACTS_DIR, f"{dataset}_{adv_i}_{reg_i_safe}.pt")

            if not os.path.exists(ckpt_base):
                st.error(f"Checkpoint not found: {ckpt_base}")
            elif not os.path.exists(ckpt_igr):
                st.error(f"Checkpoint not found: {ckpt_igr}")
            else:
                d = DATASET_DEFAULTS[dataset]
                model_name = d["model"]
                eps = d["attack_eps"]

                with st.spinner("Computing IG attributions and adversarial examples..."):
                    try:
                        fig = generate_ig_visualization(
                            dataset=dataset,
                            baseline_ckpt=ckpt_base,
                            igr_ckpt=ckpt_igr,
                            model_name=model_name,
                            baseline_label=vis_baseline,
                            igr_label=vis_igr,
                            n_samples=vis_n_samples,
                            eps=eps,
                            pgd_steps=vis_pgd_steps,
                            ig_steps=vis_ig_steps,
                            seed=vis_seed,
                        )
                        st.pyplot(fig)

                        st.markdown(
                            "**Interpretation:** If IGR works, the IG maps for clean and adversarial "
                            "inputs should look similar (columns 4 & 5), while the baseline model's "
                            "IG maps diverge under attack (columns 2 & 3)."
                        )

                        gc.collect()
                        torch.cuda.empty_cache()

                    except Exception as e:
                        st.error(f"Visualization failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())


with tab4:
    st.subheader("Apie įrankį")
    st.markdown("""
Įrankis lygina konkordacijos matus kaip IGR reguliarizatorius atribucijų atsparumui.
Atkuria ir išplečia Wang & Kong (NeurIPS 2022).

**Reguliarizatoriai:** Cosine, Pearson, DiffKendall (Zheng et al. 2023),
Soft-Spearman (Blondel et al. 2020) ir dvi kombinacijos (Cosine+DiffKendall, Cosine+Soft-Spearman).

**Priešiškasis mokymas:** AT (Madry 2018), TRADES (Zhang 2019), MART (Wang 2020).

**Vertinimas:** IFIA ataka + 5 metrikos (Top-k, Kendall τ, Spearman ρ, Cosine, Pearson);
adversarinis tikslumas (FGSM, PGD-20); cross-proxy (ataka skirtingais surogatais).
    """)

    st.subheader("CLI")
    st.code("""
# Treniruoti ir vertinti visas konfiguracijas
python run_experiment.py --dataset mnist --run-all --epochs 12

# Kombinuoti reguliarizatoriai
python run_experiment.py --dataset mnist --run-combo --epochs 12

# Tik vertinimas (be treniravimo)
python run_experiment.py --dataset mnist --run-all --eval-only

# Cross-proxy IFIA
python run_experiment.py --dataset mnist --cross-proxy

# Dashboard
python -m streamlit run dashboard.py
    """, language="bash")
