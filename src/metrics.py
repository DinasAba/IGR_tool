import torch


def topk_indices(attr, k: int):
    flat = attr.view(attr.size(0), -1)
    _, idx = torch.topk(flat.abs(), k=k, dim=1, largest=True, sorted=False)
    return idx


def topk_intersection(attr1, attr2, k: int):
    idx1 = topk_indices(attr1, k)
    idx2 = topk_indices(attr2, k)
    inter = (idx1.unsqueeze(2) == idx2.unsqueeze(1)).any(dim=2).float().sum(dim=1)
    return inter / k  # [B]


def kendall_tau(v1, v2, sample_pairs: int | None = 20000):
    # sampled poros, kad būtų greičiau; None = tikslus O(D^2)
    d = v1.numel()
    if sample_pairs is None:
        i, j = torch.triu_indices(d, d, offset=1, device=v1.device)
    else:
        i = torch.randint(0, d, (sample_pairs,), device=v1.device)
        j = torch.randint(0, d, (sample_pairs,), device=v1.device)
        mask = i != j
        i, j = i[mask], j[mask]

    a = v1[i] - v1[j]
    b = v2[i] - v2[j]
    s = torch.sign(a) * torch.sign(b)  # lygios poros -> 0
    return s.mean().item()
