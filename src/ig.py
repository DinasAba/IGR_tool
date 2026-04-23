import torch


def _estimate_chunk_size(b: int, c: int, h: int, w: int, steps: int) -> int:
    # kiek IG zingsniu tilptu GPU'iu: biudzetas ~3 GB forward+backward
    # MNIST CNN ~0.5 MB/sample, ResNet-18 CIFAR ~4 MB/sample
    pixels = c * h * w
    if pixels <= 784:  # MNIST tipo (1x28x28)
        mem_per_sample_mb = 0.5
    else:  # CIFAR tipo (3x32x32) ir didesni
        mem_per_sample_mb = 4.0
    budget_mb = 3000
    max_total = max(1, int(budget_mb / mem_per_sample_mb))
    chunk = max(1, max_total // max(b, 1))
    return min(chunk, steps)


def integrated_gradients(
    model,
    x,
    y,
    steps: int = 50,
    baseline=None,
    create_graph: bool = False,
):
    # IG su juoda bazine linija. x: [B,C,H,W], y: [B].
    # Jei netelpa viename forward'e (pvz. ResNet-18 su CIFAR), dalijam i chunk'us.
    if baseline is None:
        baseline = torch.zeros_like(x)

    b, c, h, w = x.shape
    chunk_steps = _estimate_chunk_size(b, c, h, w, steps)

    # jei viskas telpa - greitas kelias (originalus kodas)
    if chunk_steps >= steps:
        return _ig_single_pass(model, x, y, steps, baseline, create_graph)

    # chunk'uotas kelias - gradientus sumuojam per chunk'us
    alphas_all = torch.linspace(0.0, 1.0, steps, device=x.device)
    x0 = baseline
    diff = x - baseline

    grad_accum = torch.zeros_like(x)

    for start in range(0, steps, chunk_steps):
        end = min(start + chunk_steps, steps)
        alphas = alphas_all[start:end].view(-1, 1, 1, 1, 1)
        n_chunk = end - start

        x_interp = x0.unsqueeze(0) + alphas * diff.unsqueeze(0)  # [n_chunk,B,C,H,W]
        x_interp.requires_grad_(True)

        logits = model(x_interp.view(n_chunk * b, c, h, w))
        y_rep = y.repeat(n_chunk)
        selected = logits.gather(1, y_rep.view(-1, 1)).sum()

        grads = torch.autograd.grad(selected, x_interp, create_graph=create_graph)[0]
        grad_accum = grad_accum + grads.sum(dim=0)

    avg_grads = grad_accum / steps
    return diff * avg_grads


def _ig_single_pass(model, x, y, steps, baseline, create_graph):
    # greitas kelias: visi zingsniai vienu forward'u
    alphas = torch.linspace(0.0, 1.0, steps, device=x.device).view(steps, 1, 1, 1, 1)
    x0 = baseline.unsqueeze(0)
    x1 = x.unsqueeze(0)
    x_interp = x0 + alphas * (x1 - x0)
    x_interp.requires_grad_(True)

    s, b, c, h, w = x_interp.shape
    logits = model(x_interp.view(s * b, c, h, w))
    y_rep = y.repeat(s)
    selected = logits.gather(1, y_rep.view(-1, 1)).sum()

    grads = torch.autograd.grad(selected, x_interp, create_graph=create_graph)[0]
    avg_grads = grads.mean(dim=0)

    return (x - baseline) * avg_grads
