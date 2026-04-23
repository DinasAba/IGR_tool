# Konkordacijos matai atribucijoms.
# Dvi kategorijos:
#   1. Diferencijuojami REGULIARIZATORIAI (treniravimui) -> Tensor[B] nuostolis pavyzdziui
#   2. Tiksliosios VERTINIMO METRIKOS (eval)             -> skaliaras porai
# Papildomai: soft top-k intersection (IFIA atakos diferencijuojamas surogatas).
# Referencijos:
#   IGR (Wang & Kong 2022), DiffKendall (Zheng 2023), RKKD (Guan 2024), AdvAAT (Ivankay 2022).

import functools

import torch
import torch.nn.functional as F


# =================================================================
#  Diferencijuojami reguliarizatoriai (treniravimo surogatai)
#  Iejimas:  a, b - atribucijos [B, *]
#  Iseinimas: per-sample nuostolis [0, 2] intervale, Tensor[B]
# =================================================================


def cosine_reg(a: torch.Tensor, b: torch.Tensor, **_kw) -> torch.Tensor:
    # 1 - cos(a, b). IGR straipsnio pasirinkimas (Teorema 1).
    a_f = a.reshape(a.size(0), -1)
    b_f = b.reshape(b.size(0), -1)
    return 1.0 - F.cosine_similarity(a_f, b_f, dim=1, eps=1e-8)


def pearson_reg(a: torch.Tensor, b: torch.Tensor, **_kw) -> torch.Tensor:
    # 1 - Pearson(a, b). AdvAAT. Pastaba: nestabilus maziems variansams (IGR B priedas).
    a_f = a.reshape(a.size(0), -1)
    b_f = b.reshape(b.size(0), -1)
    a_c = a_f - a_f.mean(dim=1, keepdim=True)
    b_c = b_f - b_f.mean(dim=1, keepdim=True)
    return 1.0 - F.cosine_similarity(a_c, b_c, dim=1, eps=1e-8)


def diff_kendall_reg(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 20.0,
    n_pairs: int = 5000,
    **_kw,
) -> torch.Tensor:
    # 1 - tau_alpha(a, b). Diferencijuojamas Kendall per tanh aproksimacija.
    # tau_alpha = (1/K) sum_{(i,j)} tanh(alpha*(a_i - a_j)) * tanh(alpha*(b_i - b_j))
    # alpha -> stiprumas (didesnis -> arciau sign(), bet sunkesni gradientai).
    a_f = a.reshape(a.size(0), -1)
    b_f = b.reshape(b.size(0), -1)
    D = a_f.size(1)

    idx_i = torch.randint(0, D, (n_pairs,), device=a.device)
    idx_j = torch.randint(0, D, (n_pairs,), device=a.device)
    valid = idx_i != idx_j
    idx_i, idx_j = idx_i[valid], idx_j[valid]

    da = a_f[:, idx_i] - a_f[:, idx_j]          # [B, K]
    db = b_f[:, idx_i] - b_f[:, idx_j]          # [B, K]

    tau = (torch.tanh(alpha * da) * torch.tanh(alpha * db)).mean(dim=1)
    return 1.0 - tau


def soft_spearman_reg(
    a: torch.Tensor,
    b: torch.Tensor,
    beta: float = 10.0,
    max_dim: int = 200,
    **_kw,
) -> torch.Tensor:
    # 1 - soft-Spearman(a, b).
    # Soft rangai: rank_i(x) = sum_j sigma(beta*(x_i - x_j)), tada Spearman = Pearson(ranks_a, ranks_b).
    # Subsample'inam dimensijas iki max_dim del O(D^2) atminties.
    a_f = a.reshape(a.size(0), -1)
    b_f = b.reshape(b.size(0), -1)
    B, D = a_f.shape

    if D > max_dim:
        idx = torch.randperm(D, device=a.device)[:max_dim]
        a_f = a_f[:, idx]
        b_f = b_f[:, idx]

    # poriniai skirtumai -> soft rangai  [B, D, D] -> [B, D]
    rank_a = torch.sigmoid(beta * (a_f.unsqueeze(2) - a_f.unsqueeze(1))).sum(dim=2)
    rank_b = torch.sigmoid(beta * (b_f.unsqueeze(2) - b_f.unsqueeze(1))).sum(dim=2)

    # Pearson koreliacija tarp soft rangu
    rank_a_c = rank_a - rank_a.mean(dim=1, keepdim=True)
    rank_b_c = rank_b - rank_b.mean(dim=1, keepdim=True)
    rho = F.cosine_similarity(rank_a_c, rank_b_c, dim=1, eps=1e-8)
    return 1.0 - rho


# --- Kombinuotas reguliarizatorius -----------------------------------------


def combined_reg(
    a: torch.Tensor,
    b: torch.Tensor,
    reg1_fn=None,
    reg2_fn=None,
    w1: float = 0.5,
    w2: float = 0.5,
    **_kw,
) -> torch.Tensor:
    # sveriama dvieju reguliarizatoriu suma: w1*R1 + w2*R2
    # skirta sujungti vektorini (cosine/pearson) su rangu (DiffKendall/Soft-Spearman)
    loss1 = reg1_fn(a, b) if reg1_fn is not None else torch.zeros(a.size(0), device=a.device)
    loss2 = reg2_fn(a, b) if reg2_fn is not None else torch.zeros(a.size(0), device=a.device)
    return w1 * loss1 + w2 * loss2


# --- Reguliarizatoriu registras --------------------------------------------

REGULARIZERS = {
    "cosine": cosine_reg,
    "pearson": pearson_reg,
    "diff_kendall": diff_kendall_reg,
    "soft_spearman": soft_spearman_reg,
}

# Kombinuotu reguliarizatoriu preset'ai: (base_reg, rank_reg)
COMBO_PRESETS = {
    "cosine+diff_kendall": ("cosine", "diff_kendall"),
    "cosine+soft_spearman": ("cosine", "soft_spearman"),
    "pearson+diff_kendall": ("pearson", "diff_kendall"),
    "pearson+soft_spearman": ("pearson", "soft_spearman"),
}


def get_regularizer(name: str, **kwargs):
    # grazina callable(a, b) -> Tensor[B] pagal varda.
    # Palaiko pavienius ir kombinuotus: get_regularizer("cosine+diff_kendall", w1=0.5, w2=0.5).
    # Papildomi kwargs pririsami per functools.partial.
    if name in COMBO_PRESETS:
        r1_name, r2_name = COMBO_PRESETS[name]
        w1 = kwargs.pop("w1", 0.5)
        w2 = kwargs.pop("w2", 0.5)
        # sub-reguliarizatoriai su savais kwargs (prefiksai r1_ ir r2_)
        r1_kwargs = {}
        r2_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("r1_"):
                r1_kwargs[k[3:]] = v
            elif k.startswith("r2_"):
                r2_kwargs[k[3:]] = v
        reg1 = functools.partial(REGULARIZERS[r1_name], **r1_kwargs) if r1_kwargs else REGULARIZERS[r1_name]
        reg2 = functools.partial(REGULARIZERS[r2_name], **r2_kwargs) if r2_kwargs else REGULARIZERS[r2_name]
        return functools.partial(combined_reg, reg1_fn=reg1, reg2_fn=reg2, w1=w1, w2=w2)

    if name not in REGULARIZERS:
        raise ValueError(
            f"Unknown regularizer '{name}'. Choose from: "
            f"{list(REGULARIZERS.keys()) + list(COMBO_PRESETS.keys())}"
        )
    fn = REGULARIZERS[name]
    return functools.partial(fn, **kwargs) if kwargs else fn


# =================================================================
#  Soft top-k intersection (IFIA atakos diferencijuojamas surogatas)
# =================================================================


def soft_topk_intersection(
    attr1: torch.Tensor,
    attr2: torch.Tensor,
    k: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    # diferencijuojama top-k intersection aproksimacija per sigmoid kauke apie k-taji didziausia |attr|
    # grazina [B], ~[0, 1]
    a1 = attr1.reshape(attr1.size(0), -1).abs()
    a2 = attr2.reshape(attr2.size(0), -1).abs()

    # k-tasis didziausias = slenkstis (sort yra diferencijuojamas PyTorch'e)
    thresh1 = a1.sort(dim=1, descending=True).values[:, k - 1 : k]   # [B,1]
    thresh2 = a2.sort(dim=1, descending=True).values[:, k - 1 : k]

    mask1 = torch.sigmoid((a1 - thresh1) / temperature)               # [B,D]
    mask2 = torch.sigmoid((a2 - thresh2) / temperature)

    return (mask1 * mask2).sum(dim=1) / k


def cosine_sim_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # per-sample cos, [B]. IFIA proxy.
    a_f = a.reshape(a.size(0), -1)
    b_f = b.reshape(b.size(0), -1)
    return F.cosine_similarity(a_f, b_f, dim=1, eps=1e-8)


def pearson_sim_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # per-sample Pearson, [B]. IFIA proxy.
    a_f = a.reshape(a.size(0), -1)
    b_f = b.reshape(b.size(0), -1)
    a_c = a_f - a_f.mean(dim=1, keepdim=True)
    b_c = b_f - b_f.mean(dim=1, keepdim=True)
    return F.cosine_similarity(a_c, b_c, dim=1, eps=1e-8)


def diff_kendall_sim_batch(
    a: torch.Tensor, b: torch.Tensor,
    alpha: float = 20.0, n_pairs: int = 5000,
) -> torch.Tensor:
    # per-sample diferencijuojamas Kendall, [B]. IFIA proxy.
    a_f = a.reshape(a.size(0), -1)
    b_f = b.reshape(b.size(0), -1)
    D = a_f.size(1)
    idx_i = torch.randint(0, D, (n_pairs,), device=a.device)
    idx_j = torch.randint(0, D, (n_pairs,), device=a.device)
    valid = idx_i != idx_j
    idx_i, idx_j = idx_i[valid], idx_j[valid]
    da = a_f[:, idx_i] - a_f[:, idx_j]
    db = b_f[:, idx_i] - b_f[:, idx_j]
    return (torch.tanh(alpha * da) * torch.tanh(alpha * db)).mean(dim=1)


def soft_spearman_sim_batch(
    a: torch.Tensor, b: torch.Tensor,
    beta: float = 10.0, max_dim: int = 200,
) -> torch.Tensor:
    # per-sample soft Spearman, [B]. IFIA proxy.
    a_f = a.reshape(a.size(0), -1)
    b_f = b.reshape(b.size(0), -1)
    B, D = a_f.shape
    if D > max_dim:
        idx = torch.randperm(D, device=a.device)[:max_dim]
        a_f = a_f[:, idx]
        b_f = b_f[:, idx]
    rank_a = torch.sigmoid(beta * (a_f.unsqueeze(2) - a_f.unsqueeze(1))).sum(dim=2)
    rank_b = torch.sigmoid(beta * (b_f.unsqueeze(2) - b_f.unsqueeze(1))).sum(dim=2)
    rank_a_c = rank_a - rank_a.mean(dim=1, keepdim=True)
    rank_b_c = rank_b - rank_b.mean(dim=1, keepdim=True)
    return F.cosine_similarity(rank_a_c, rank_b_c, dim=1, eps=1e-8)


# IFIA proxy funkciju registras: name -> callable(a, b) -> Tensor[B]
IFIA_PROXIES = {
    "soft_topk": None,   # atskirai, reikia k ir temperature
    "cosine": cosine_sim_batch,
    "pearson": pearson_sim_batch,
    "diff_kendall": diff_kendall_sim_batch,
    "soft_spearman": soft_spearman_sim_batch,
}


# =================================================================
#  Tikslios vertinimo metrikos (nediferencijuojamos, naudojam eval)
#  Iejimas:  v1, v2 - ploksti 1-D tensoriai [D]
#  Iseinimas: float skaliaras
# =================================================================


def kendall_tau_exact(
    v1: torch.Tensor, v2: torch.Tensor, sample_pairs: int = 20000, **_kw,
) -> float:
    # Kendall tau-a per poru zenklu sutapima (galimas sub-sampling)
    d = v1.numel()
    if sample_pairs is None or sample_pairs >= d * (d - 1) // 2:
        i, j = torch.triu_indices(d, d, offset=1, device=v1.device)
    else:
        i = torch.randint(0, d, (sample_pairs,), device=v1.device)
        j = torch.randint(0, d, (sample_pairs,), device=v1.device)
        valid = i != j
        i, j = i[valid], j[valid]

    da = v1[i] - v1[j]
    db = v2[i] - v2[j]
    return (torch.sign(da) * torch.sign(db)).mean().item()


def spearman_rho_exact(v1: torch.Tensor, v2: torch.Tensor, **_kw) -> float:
    # tiksli Spearman rho = Pearson(rank(v1), rank(v2))
    def _rank(v: torch.Tensor) -> torch.Tensor:
        order = v.argsort()
        rank = torch.empty_like(v)
        rank[order] = torch.arange(len(v), device=v.device, dtype=v.dtype)
        return rank

    r1 = _rank(v1)
    r2 = _rank(v2)
    r1c = r1 - r1.mean()
    r2c = r2 - r2.mean()
    return F.cosine_similarity(r1c.unsqueeze(0), r2c.unsqueeze(0)).item()


def cosine_sim_exact(v1: torch.Tensor, v2: torch.Tensor, **_kw) -> float:
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def pearson_exact(v1: torch.Tensor, v2: torch.Tensor, **_kw) -> float:
    v1c = v1 - v1.mean()
    v2c = v2 - v2.mean()
    return F.cosine_similarity(v1c.unsqueeze(0), v2c.unsqueeze(0)).item()


def topk_intersection_exact(
    attr1: torch.Tensor, attr2: torch.Tensor, k: int, **_kw
) -> float:
    # tikslus (diskretus) top-k intersection dviem ploksciam vektoriam
    _, idx1 = attr1.abs().topk(k)
    _, idx2 = attr2.abs().topk(k)
    # CPU set intersection - kad broadcast nenustumtu GPU atminties
    set1 = set(idx1.cpu().tolist())
    set2 = set(idx2.cpu().tolist())
    return len(set1 & set2) / k


# tvarkinga tvarka lenteliu stulpeliams
EVAL_METRICS = {
    "top_k": None,              # atskirai, reikia k
    "kendall_tau": kendall_tau_exact,
    "spearman_rho": spearman_rho_exact,
    "cosine_sim": cosine_sim_exact,
    "pearson": pearson_exact,
}
