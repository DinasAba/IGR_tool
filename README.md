# IGR Tool: Concordance Measure Comparison for Attribution Protection

Replication and extension of:

> Fan Wang, Adams Wai-Kin Kong. *"Exploiting the Relationship Between Kendall's Rank Correlation and Cosine Similarity for Attribution Protection"*. NeurIPS 2022.

## Setup

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Reproduce Table 2 (MNIST, 90 epochs)

```bash
python run_experiment.py --dataset mnist --run-all --epochs 90
```

This trains 16 models (4 adversarial methods x 4 regularizers + baseline) and evaluates each under IFIA attack. Results are saved to `results/mnist/`.

### Quick test run (3 epochs, fast eval)

```bash
python run_experiment.py --dataset mnist --run-all --epochs 3 --quick
```

### Evaluate existing models only (no training)

```bash
python run_experiment.py --dataset mnist --run-all --eval-only
```

### Interactive menu

```bash
python interactive.py
```

### Streamlit dashboard

```bash
python -m streamlit run dashboard.py
```

## Project Structure

```
igr_tool/
  src/
    models.py          # MNIST CNN (4conv+3FC), CIFAR-10 ResNet-18
    datasets.py         # MNIST, Fashion-MNIST, CIFAR-10 loaders
    losses.py           # AT, TRADES, MART losses + PGD attack
    train.py            # Training loop
    ig.py               # Integrated Gradients (baseline=0)
    concordance.py      # Regularizers + exact evaluation metrics
    eval_ifia.py        # IFIA attack + evaluation pipeline
    eval_adversarial.py # White-box adversarial accuracy (Section 6.3)
    eval_activation.py  # Activation consistency (Section 6.4)
    utils.py            # Seeds, device, helpers
  tests/
    test_metrics.py     # Unit tests for all concordance metrics
  run_experiment.py     # CLI entry point (argparse)
  interactive.py        # Interactive text-UI menu
  dashboard.py          # Streamlit dashboard with charts
  requirements.txt
```

## CLI Reference

```
python run_experiment.py --dataset {mnist,fashion_mnist,cifar10}
    --adv {none,at,trades,mart}
    --reg {none,cosine,diff_kendall,soft_spearman,pearson}
    --epochs N
    --seed N
    --eval-only          # skip training, evaluate existing checkpoint
    --run-all            # train+eval all 16 combinations
    --quick              # fast eval (50 steps, 1 restart, 50 samples)
    --fast-train         # fast training (PGD steps=3)
    --lambda-igr FLOAT   # override lambda for IGR regularizer
    --eval-adv           # also run adversarial accuracy eval
    --eval-activation    # also run activation consistency eval
    --export DIR         # output directory (default: results/)
```

## Paper Replication Details

### Hyperparameters (MNIST)

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | 4 conv + 3 FC | Paper Section 6 |
| Optimizer | Adam, lr=1e-4 | Paper Section 6 |
| Epochs | 90 | Paper Section 6 |
| Epsilon (L-inf) | 0.3 | Paper Section 6 |
| PGD steps (train) | 7 | Paper Appendix (Algorithm 1 structure) |
| PGD step-size (train) | eps/4 = 0.075 | Standard practice |
| IFIA steps (eval) | 200 | Paper Section 5 |
| IFIA restarts | 5 | Paper Section 5 |
| IFIA k | 100 | Paper Section 5 |
| IG steps (eval) | 50 | Configurable |
| Lambda IGR | 1.0 | Not specified in paper; using 1.0 as default |

### Evaluation Metrics

**Paper metrics (Table 2):**

| Metric | Definition | Notes |
|--------|-----------|-------|
| Top-k intersection | \|topk(abs(IG_clean)) ∩ topk(abs(IG_adv))\| / k | k=100, measures overlap of most important features |
| Kendall tau | Pairwise sign agreement over feature rankings | tau-a variant; ties contribute 0 (see below) |

**Extension metrics (our additions for concordance comparison):**

| Metric | Definition | Notes |
|--------|-----------|-------|
| Spearman rho | Pearson correlation of rank vectors | Exact ranking, no ties |
| Cosine similarity | cos(IG_clean, IG_adv) | Also used as the IGR regularizer |
| Pearson r | Pearson correlation of raw attribution vectors | Sensitive to mean shift |

### Tie Handling in Kendall Tau

We use **Kendall tau-a** (not tau-b):

```
tau = mean(sign(a_i - a_j) * sign(b_i - b_j))  for all pairs (i, j)
```

When values are tied (`a_i == a_j`), `sign(0) = 0`, so tied pairs contribute 0 to the sum. This is consistent with the standard tau-a definition. Tau-b adjusts the denominator for ties, but since attribution vectors are continuous (float), ties are extremely rare in practice.

For efficiency, we subsample pairs (`sample_pairs=10000` by default for eval, `sample_pairs=20000` available). Subsampling introduces small variance but is necessary for large feature spaces (MNIST: 784 features = 306,936 pairs).

### Differences from the Paper

1. **Lambda (IGR weight):** The paper does not specify the lambda value for the IGR regularizer. We use lambda=1.0 for all methods. The paper states: "we keep the hyper-parameters the same for models with or without IGR."

2. **PGD training parameters:** The paper does not explicitly state the number of PGD steps or step size used during training. We use 7 steps with step_size=eps/4 based on Algorithm 1 structure and standard practice.

3. **IFIA proxy:** The paper uses cosine similarity as the differentiable proxy for the IFIA attack (justified by Theorem 1: Kendall tau upper-bounded by cosine distance). Our implementation supports both `cosine` (default, matching the paper) and `soft_topk` proxy.

4. **Additional metrics:** The paper evaluates only top-k intersection and Kendall tau. We additionally compute Spearman rho, cosine similarity, and Pearson correlation for broader concordance comparison. These are clearly labeled as extensions.

5. **Additional regularizers:** Beyond the paper's cosine regularizer (IGR), we implement differentiable Kendall approximation (DiffKendall), soft Spearman, and Pearson as alternative regularizers.

## Training Objectives

| Method | Training Loss | PGD Attack Loss |
|--------|-------------|----------------|
| Standard | CE(f(x), y) | N/A |
| AT | CE(f(x_adv), y) | CE |
| TRADES | CE(f(x),y) + beta*KL(f(x)\|\|f(x_adv)) | KL |
| MART | BCE(f(x_adv),y) + beta*KL*(1-f_y(x)) | CE |

With IGR: `total_loss = robust_loss + lambda * (1 - regularizer(IG(x), IG(x_adv)))`

## Regularizers

| Name | Formula | Reference |
|------|---------|-----------|
| cosine | 1 - cos(a, b) | IGR paper (Theorem 1) |
| pearson | 1 - Pearson(a, b) | AdvAAT (Ivankay et al., 2022) |
| diff_kendall | 1 - tau_alpha(a, b) via tanh approx | DiffKendall (Zheng et al., 2023) |
| soft_spearman | 1 - Spearman(soft_rank(a), soft_rank(b)) | Sigmoid-based soft ranking |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Expected Runtime (MNIST, single GPU)

| Task | Time |
|------|------|
| Train 1 model (90 epochs) | ~15-30 min |
| Train all 16 models (90 epochs) | ~4-8 hours |
| IFIA eval, 1 model (200 steps, 5 restarts) | ~10-20 min |
| IFIA eval, 1 model (quick: 50 steps, 1 restart) | ~2-3 min |
| Full pipeline (16 models, train+eval, quick) | ~1-2 hours |
| Full pipeline (16 models, train+eval, 90 epochs) | ~8-12 hours |

Times measured on NVIDIA GPU with ~6GB VRAM. CPU will be significantly slower.

## Output Files

```
results/<dataset>/
  results.json          # IFIA metrics (top-k, kendall, spearman, cosine, pearson)
  results.csv           # Same in CSV format
  results.tex           # LaTeX table (booktabs)
  adv_accuracy.json     # Adversarial accuracy (if --eval-adv)
  activation_consistency.json  # Activation consistency (if --eval-activation)

artifacts/
  <dataset>_<adv>_<reg>.pt   # Model checkpoints
```
