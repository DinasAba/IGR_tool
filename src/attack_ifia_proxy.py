# Proxy IFIA ataka: maksimizuoja atribucijos nepanasuma, baudzia klasifikacijos nukrypima
import torch
import torch.nn.functional as F

from .ig import integrated_gradients


def cosine_similarity(a, b, eps=1e-8):
    a = a.view(a.size(0), -1)
    b = b.view(b.size(0), -1)
    an = a.norm(dim=1).clamp_min(eps)
    bn = b.norm(dim=1).clamp_min(eps)
    return (a * b).sum(dim=1) / (an * bn)


def _attr_for_similarity(attr: torch.Tensor, use_abs_attr: bool) -> torch.Tensor:
    return attr.abs() if use_abs_attr else attr


def ifia_proxy_attack(
    model,
    x,
    y,
    eps: float,
    steps: int,
    step_size: float,
    restarts: int,
    k: int,
    ig_steps_attack: int,
    cls_weight: float = 1.0,
    use_abs_attr: bool = False,
):
    del k  # paliktas parase del suderinamumo su config
    model.eval()

    ig_x = integrated_gradients(model, x, y, steps=ig_steps_attack, create_graph=False)
    ig_x = _attr_for_similarity(ig_x, use_abs_attr)

    # kiekvienam restart'ui grazinam savo adv, kad galetume suvidurkinti metrikas
    adv_list = []

    for _ in range(restarts):
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1).detach()

        for _ in range(steps):
            x_adv.requires_grad_(True)

            ig_adv = integrated_gradients(model, x_adv, y, steps=ig_steps_attack, create_graph=False)
            ig_adv = _attr_for_similarity(ig_adv, use_abs_attr)

            # diferencijuojamas surogatas: 1 - cos(IG(x), IG(x_adv))
            cos = cosine_similarity(ig_x, ig_adv).mean()
            dissim = 1.0 - cos

            logits = model(x_adv)
            ce = F.cross_entropy(logits, y)

            obj = dissim - cls_weight * ce
            grad = torch.autograd.grad(obj, x_adv)[0]

            with torch.no_grad():
                x_adv = x_adv + step_size * torch.sign(grad)
                x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)  # L_inf projekcija
                x_adv = x_adv.clamp(0, 1).detach()

        adv_list.append(x_adv)

    return adv_list


def ifia_proxy_train_attack(
    model,
    x,
    y,
    eps: float,
    steps: int,
    step_size: float,
    ig_steps_attack: int,
    cls_weight: float = 1.0,
    use_abs_attr: bool = False,
):
    # paprastesne vieno restart'o ataka - naudojama x_tilde generavimui IGR treniravime
    was_training = model.training
    model.eval()

    ig_x = integrated_gradients(model, x, y, steps=ig_steps_attack, create_graph=False)
    ig_x = _attr_for_similarity(ig_x, use_abs_attr)

    x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        ig_adv = integrated_gradients(model, x_adv, y, steps=ig_steps_attack, create_graph=False)
        ig_adv = _attr_for_similarity(ig_adv, use_abs_attr)

        cos = cosine_similarity(ig_x, ig_adv).mean()
        dissim = 1.0 - cos

        logits = model(x_adv)
        ce = F.cross_entropy(logits, y)

        obj = dissim - cls_weight * ce
        grad = torch.autograd.grad(obj, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + step_size * torch.sign(grad)
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0, 1).detach()

    if was_training:
        model.train()
    else:
        model.eval()

    return x_adv
