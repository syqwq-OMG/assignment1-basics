from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def learning_rate_schedule(t:int, a_max, a_min, T_w, T_c):
    if t < T_w:
        return a_max * t / T_w
    elif t < T_c:
        return a_min + (1+math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (a_max - a_min) / 2
    else:
        return a_min

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, lam=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, eps, lam = group["lr"], group["eps"], group["lam"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0

                t = state["t"] + 1
                m = state["m"]
                v = state["v"]
                grad = p.grad

                # Adjusted learning rate
                a_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Decoupled weight decay
                p.mul_(1 - lr * lam)

                # First moment
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Second moment
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Parameter update
                p.addcdiv_(m, v.sqrt().add(eps), value=-a_t)

                state["t"] = t

        return loss

    def my_step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, eps, lam = group["lr"], group["eps"], group["lam"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                t = state.get("t", 0)
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                a_t = lr * math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))
                p.data -= lr * lam * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                p.data -= a_t * m / (torch.sqrt(v) + eps)

                state["t"], state["m"], state["v"] = t + 1, m, v
        return loss
