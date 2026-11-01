import math


def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    n_warmup=1000,
    init_div_factor=100.0,
    fract_decay=0.1,
    decay_type="linear",
):
    n_anneal = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        if step < n_hold:
            return 1.0
        if step >= n_iterations or n_anneal == 0:
            return final_lr_factor
        progress = (step - n_hold) / n_anneal
        if decay_type == "linear":
            return final_lr_factor + (1 - final_lr_factor) * (1 - progress)
        if decay_type == "cosine":
            return final_lr_factor + (1 - final_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))
        if decay_type == "exp":
            return final_lr_factor ** progress
        raise ValueError(f"decay type {decay_type} is not in ['linear', 'cosine', 'exp']")

    return schedule


def cosine_schedule(n_iterations, n_warmup, min_factor):
    def schedule(step):
        if step < n_warmup:
            return (step + 1) / max(n_warmup, 1)
        if step >= n_iterations:
            return min_factor
        progress = (step - n_warmup) / max(n_iterations - n_warmup, 1)
        coeff = 0.5 * (1 + math.cos(math.pi * progress))
        return min_factor + coeff * (1 - min_factor)

    return schedule