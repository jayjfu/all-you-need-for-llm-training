import torch


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                ortho = newton_schulz(update, steps=group["ns_steps"])
                scaled_lr = group["lr"] * max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(ortho, alpha=-scaled_lr)


def split_params(model):
    matrix, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Matrices go to Muon, embedding and lm_head stay on AdamW
        if p.ndim == 2 and not name.endswith("embed_tokens.weight") and not name.endswith("lm_head.weight"):
            matrix.append(p)
        else:
            adam.append(p)
    return matrix, adam


def build_optimizer(model, matrix_lr=0.02, embed_lr=3e-3, weight_decay=0.0):
    matrix, adam = split_params(model)
    muon = Muon(matrix, lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(adam, lr=embed_lr, betas=(0.9, 0.95), weight_decay=0.0)
    return muon, adamw