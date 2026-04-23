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

### Reproduce Table 2 (MNIST, 12 epochs)

```bash
python run_experiment.py --dataset mnist --run-all --epochs 12
```

This trains 16 models (4 adversarial methods x 4 regularizers + baseline) and evaluates each under IFIA attack. Results are saved to `results/mnist/`.

### Run combined regularizers (cosine+diff_kendall, cosine+soft_spearman)

```bash
python run_experiment.py --dataset mnist --run-combo --epochs 12
```

Trains 6 additional models (AT/TRADES/MART x 2 combo regularizers) with weighted combinations of a vector-based and a rank-based regularizer. Weights default to `w1=w2=0.5` and can be overridden:

```bash
python run_experiment.py --dataset mnist --run-combo --epochs 12 --combo-w1 0.7 --combo-w2 0.3
```

Results are saved to `results/<dataset>/combo_results.json`.

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
    --run-combo          # train+eval combined regularizers (cosine+diff_kendall, cosine+soft_spearman)
    --combo-w1 FLOAT     # weight for first (vector) regularizer in combo (default: 0.5)
    --combo-w2 FLOAT     # weight for second (rank) regularizer in combo (default: 0.5)
    --quick              # fast eval (50 steps, 1 restart, 50 samples)
    --fast-train         # fast training (PGD steps=3)
    --lambda-igr FLOAT   # override lambda for IGR regularizer
    --eval-adv           # also run adversarial accuracy eval
    --eval-activation    # also run activation consistency eval
    --export DIR         # output directory (default: results/)
```

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
| Train 1 model (12 epochs) | ~2-4 min |
| Train all 16 models (12 epochs) | ~30-60 min |
| IFIA eval, 1 model (200 steps, 5 restarts) | ~10-20 min |
| IFIA eval, 1 model (quick: 50 steps, 1 restart) | ~2-3 min |
| Full pipeline (16 models, train+eval, quick) | ~1-2 hours |
| Full pipeline (16 models, train+eval, 12 epochs) | ~2-3 hours |

Times measured on NVIDIA GPU with ~6GB VRAM. CPU will be significantly slower.

## Output Files

```
results/<dataset>/
  results.json          # IFIA metrics (top-k, kendall, spearman, cosine, pearson)
  results.csv           # Same in CSV format
  results.tex           # LaTeX table (booktabs)
  adv_accuracy.json     # Adversarial accuracy (if --eval-adv)
  activation_consistency.json  # Activation consistency (if --eval-activation)
  combo_results.json    # Combined regularizer metrics (if --run-combo)
  combo_adv_accuracy.json  # Combined regularizer adv accuracy (if --run-combo --eval-adv)

artifacts/
  <dataset>_<adv>_<reg>.pt   # Model checkpoints
```
