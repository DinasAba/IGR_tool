# Priesiskojo mokymo nuostoliai (AT, TRADES, MART) ir PGD ataka.
# Referencijos: AT (Madry 2018), TRADES (Zhang 2019), MART (Wang 2020).

import torch
import torch.nn.functional as F


# =================================================================
# PGD ataka (standartine, be IG - greita)
# =================================================================

@torch.no_grad()
def _pgd_init(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    return x_adv.clamp(0, 1)


def pgd_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    attack_loss: str = "ce",
) -> torch.Tensor:
    # PGD adv pavyzdziai be IG. attack_loss: "ce" AT/MART'ui, "kl" TRADES'ui.
    was_training = model.training
    model.eval()

    # KL atakai reikia natural tikimybiu
    nat_probs = None
    if attack_loss == "kl":
        with torch.no_grad():
            nat_probs = F.softmax(model(x), dim=1)

    x_adv = _pgd_init(x, eps)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)

        if attack_loss == "ce":
            loss = F.cross_entropy(logits, y)
        elif attack_loss == "kl":
            loss = F.kl_div(
                F.log_softmax(logits, dim=1), nat_probs, reduction="batchmean"
            )
        else:
            raise ValueError(f"Unknown attack_loss: {attack_loss}")

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)

    if was_training:
        model.train()
    return x_adv.detach()


# =================================================================
# Nuostolio funkcijos
# =================================================================

def standard_loss(model, x, x_adv, y, **kwargs):
    # standartinis CE ant svariu - baselinas
    return F.cross_entropy(model(x), y)


def at_loss(model, x, x_adv, y, **kwargs):
    # AT: CE(f(x_adv), y)
    return F.cross_entropy(model(x_adv), y)


def trades_loss(model, x, x_adv, y, beta: float = 6.0, **kwargs):
    # TRADES: CE(f(x), y) + beta * KL(softmax(f(x)) || softmax(f(x_adv)))
    # NaN apsauga: KL clamp'inam, kad nesprogtu gradientai su aukstu lr (SGD 0.1)
    # ir dideliu beta treniravimo pradzioje.
    logits_nat = model(x)
    logits_adv = model(x_adv)
    ce = F.cross_entropy(logits_nat, y)

    # natural tikimybiu detach - stabilizuoja gradientus (kaip MART'e)
    # gradientai teka tik per adv saka KL nariui
    nat_probs = F.softmax(logits_nat, dim=1).detach()
    kl = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        nat_probs,
        reduction="batchmean",
    )
    # clamp - apsauga nuo ekstremaliu reiksmiu su atsitiktiniais svoriais treniravimo pradzioje
    kl = torch.clamp(kl, max=50.0)
    return ce + beta * kl


def mart_loss(model, x, x_adv, y, beta: float = 6.0, **kwargs):
    # MART: BCE(f(x_adv), y) + beta * KL(f(x) || f(x_adv)) * (1 - f_y(x))
    logits_nat = model(x)
    logits_adv = model(x_adv)

    probs_adv = F.softmax(logits_adv, dim=1)
    probs_nat = F.softmax(logits_nat, dim=1)

    # boosted CE: -log p_y(x_adv) - log(1 - p_y(x_adv))
    bce = (
        F.cross_entropy(logits_adv, y, reduction="none")
        + F.nll_loss(
            torch.log(1.0001 - probs_adv + 1e-12), y, reduction="none"
        )
    )

    # svorinis KL: (1 - p_y(x)) * KL
    fy = probs_nat.gather(1, y.unsqueeze(1)).squeeze(1).detach()
    kl = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        probs_nat.detach(),
        reduction="none",
    ).sum(dim=1)

    return bce.mean() + beta * (kl * (1.0 - fy)).mean()


# --- Registras ---

LOSSES = {
    "none": standard_loss,
    "at": at_loss,
    "trades": trades_loss,
    "mart": mart_loss,
}

# kurio PGD nuostolio imam x_adv generavimui
ADV_ATTACK_LOSS = {
    "none": "ce",
    "at": "ce",
    "trades": "kl",
    "mart": "ce",
}
