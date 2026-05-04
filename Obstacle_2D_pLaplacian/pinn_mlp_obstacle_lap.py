import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# MLP-PINN for p-Laplacian obstacle problem
# Domain: Ω = [0,2]^2
#
# PDE inequality:
#   -div(|∇u|^{p-2} ∇u) + 1 >= 0
#
# Obstacle:
#   psi(x,y) = 1,  if 0.5 <= x <= 1.5
#            = 0,  otherwise
#
# p = 4
#
# Exact solution:
#   u(x,y) = 3/4 |x + 7.75086|^{4/3} - 11.50434,     x < 0.5
#          = 1,                                        0.5 <= x <= 1.5
#          = 3/4 |-x + 9.75086|^{4/3} - 11.50434,    x > 1.5
# ============================================================


def set_seed(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 12345

    # Domain
    xmin: float = 0.0
    xmax: float = 2.0
    ymin: float = 0.0
    ymax: float = 2.0

    # p-Laplacian
    p: float = 4.0
    eps_grad: float = 1e-8

    # MLP model
    widths: tuple = (2, 64, 64, 64, 1)
    activation: str = "tanh"

    # Training
    epochs: int = 10_000
    lr: float = 1e-3
    n_interior: int = 2048
    n_boundary: int = 1024
    print_every: int = 250

    # Scheduler
    scheduler_step: int = 2000
    scheduler_gamma: float = 0.5

    # Loss weights
    lambda_obs: float = 25.0
    lambda_pde: float = 1.0
    lambda_comp: float = 1.0
    lambda_bc: float = 1e4

    # Evaluation
    n_eval_2d: int = 256

    # Output
    out_dir: str = "outputs_mlp_plap_obstacle_exact"


# ------------------------------------------------------------
# Problem definition
# ------------------------------------------------------------

def psi_fn(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0]

    psi = torch.zeros_like(x1)
    inside = (x1 >= 0.5) & (x1 <= 1.5)
    psi = torch.where(inside, torch.ones_like(x1), psi)

    return psi


def u_exact_fn(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0]

    left = 0.75 * torch.abs(x1 + 7.75086) ** (4.0 / 3.0) - 11.50434
    middle = torch.ones_like(x1)
    right = 0.75 * torch.abs(-x1 + 9.75086) ** (4.0 / 3.0) - 11.50434

    u = torch.where(x1 < 0.5, left, middle)
    u = torch.where(x1 > 1.5, right, u)

    return u


def g_fn(x: torch.Tensor) -> torch.Tensor:
    return u_exact_fn(x)


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

def make_interior_points(cfg: Config) -> torch.Tensor:
    x1 = cfg.xmin + (cfg.xmax - cfg.xmin) * torch.rand(
        cfg.n_interior, 1, device=cfg.device, dtype=cfg.dtype
    )
    x2 = cfg.ymin + (cfg.ymax - cfg.ymin) * torch.rand(
        cfg.n_interior, 1, device=cfg.device, dtype=cfg.dtype
    )

    return torch.cat([x1, x2], dim=1)


def make_boundary_points(cfg: Config) -> torch.Tensor:
    n = cfg.n_boundary
    device, dtype = cfg.device, cfg.dtype

    s = torch.rand(n, 1, device=device, dtype=dtype)
    side = torch.randint(0, 4, (n, 1), device=device)

    x = torch.zeros(n, 2, device=device, dtype=dtype)

    # x = xmin
    mask = side[:, 0] == 0
    x[mask, 0] = cfg.xmin
    x[mask, 1] = cfg.ymin + (cfg.ymax - cfg.ymin) * s[mask, 0]

    # x = xmax
    mask = side[:, 0] == 1
    x[mask, 0] = cfg.xmax
    x[mask, 1] = cfg.ymin + (cfg.ymax - cfg.ymin) * s[mask, 0]

    # y = ymin
    mask = side[:, 0] == 2
    x[mask, 0] = cfg.xmin + (cfg.xmax - cfg.xmin) * s[mask, 0]
    x[mask, 1] = cfg.ymin

    # y = ymax
    mask = side[:, 0] == 3
    x[mask, 0] = cfg.xmin + (cfg.xmax - cfg.xmin) * s[mask, 0]
    x[mask, 1] = cfg.ymax

    return x


# ------------------------------------------------------------
# MLP model
# ------------------------------------------------------------

def get_activation(name: str) -> nn.Module:
    name = name.lower()

    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"Unknown activation: {name}")


class MLPNet(nn.Module):
    def __init__(self, widths, activation: str = "tanh"):
        super().__init__()

        act = get_activation(activation)

        layers = []
        for din, dout in zip(widths[:-2], widths[1:-1]):
            layers.append(nn.Linear(din, dout))
            layers.append(act.__class__())

        layers.append(nn.Linear(widths[-2], widths[-1]))

        self.net = nn.Sequential(*layers)

        self.initialize()

    def initialize(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquareObstacleMLPPINN(nn.Module):
    def __init__(self, widths, activation: str = "tanh"):
        super().__init__()
        self.mlp = MLPNet(widths=widths, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
        


# ------------------------------------------------------------
# p-Laplacian operator
# ------------------------------------------------------------

def p_laplace_operator(model: nn.Module, x: torch.Tensor, cfg: Config):
    x = x.clone().detach().requires_grad_(True)

    u = model(x)

    grad_u = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    grad_norm_sq = torch.sum(grad_u**2, dim=1, keepdim=True)
    weight = (grad_norm_sq + cfg.eps_grad) ** ((cfg.p - 2.0) / 2.0)

    flux = weight * grad_u

    div_flux = torch.zeros_like(u)

    for j in range(2):
        flux_j = flux[:, j:j + 1]

        d_flux_j = torch.autograd.grad(
            flux_j,
            x,
            grad_outputs=torch.ones_like(flux_j),
            create_graph=True,
        )[0][:, j:j + 1]

        div_flux = div_flux + d_flux_j

    A_u = -div_flux + 1.0

    return x, u, grad_u, A_u


# ------------------------------------------------------------
# Error diagnostics
# ------------------------------------------------------------

def relative_l2_error(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(
        (torch.linalg.norm(pred - truth) / torch.linalg.norm(truth))
        .detach()
        .cpu()
    )


def relative_linf_error(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(
        (torch.max(torch.abs(pred - truth)) / torch.max(torch.abs(truth)))
        .detach()
        .cpu()
    )


@torch.no_grad()
def evaluate_on_square(model: nn.Module, cfg: Config):
    n = cfg.n_eval_2d

    x1 = torch.linspace(cfg.xmin, cfg.xmax, n, device=cfg.device, dtype=cfg.dtype)
    x2 = torch.linspace(cfg.ymin, cfg.ymax, n, device=cfg.device, dtype=cfg.dtype)

    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    pts = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

    U_pred = model(pts).reshape(n, n)
    U_true = u_exact_fn(pts).reshape(n, n)
    ERR = torch.abs(U_pred - U_true)

    rel_l2 = relative_l2_error(U_pred.reshape(-1, 1), U_true.reshape(-1, 1))
    rel_linf = relative_linf_error(U_pred.reshape(-1, 1), U_true.reshape(-1, 1))

    return {
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
        "max_abs_error": float(torch.max(ERR).detach().cpu()),
    }


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train(cfg: Config):
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", cfg.device)
    if cfg.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = SquareObstacleMLPPINN(
        widths=cfg.widths,
        activation=cfg.activation,
    ).to(cfg.device, cfg.dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step,
        gamma=cfg.scheduler_gamma,
    )

    history = {
        "total": [],
        "obs": [],
        "pde": [],
        "comp": [],
        "bc": [],
        "rel_l2": [],
        "rel_linf": [],
        "max_abs_error": [],
        "lr": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()

        x_interior = make_interior_points(cfg)

        x_req, u, _, A_u = p_laplace_operator(model, x_interior, cfg)

        psi = psi_fn(x_req).view(-1, 1)

        # Obstacle constraint: u >= psi
        obs_violation = torch.relu(psi - u)

        # PDE inequality: A(u) >= 0
        pde_violation = torch.relu(-A_u)

        # Complementarity condition: (u - psi) A(u) = 0
        comp_term = (u - psi) * A_u

        loss_obs = torch.mean(obs_violation**2)
        loss_pde = torch.mean(pde_violation**2)
        loss_comp = torch.mean(comp_term**2)

        x_boundary = make_boundary_points(cfg)
        u_boundary = model(x_boundary)
        g_boundary = g_fn(x_boundary).view(-1, 1)
        loss_bc = torch.mean((u_boundary - g_boundary) ** 2)

        loss = (
            cfg.lambda_obs * loss_obs
            + cfg.lambda_pde * loss_pde
            + cfg.lambda_comp * loss_comp
            + cfg.lambda_bc * loss_bc
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % cfg.print_every == 0 or epoch == 1:
            eval_data = evaluate_on_square(model, cfg)
            current_lr = optimizer.param_groups[0]["lr"]

            history["total"].append(float(loss.detach().cpu()))
            history["obs"].append(float(loss_obs.detach().cpu()))
            history["pde"].append(float(loss_pde.detach().cpu()))
            history["comp"].append(float(loss_comp.detach().cpu()))
            history["bc"].append(float(loss_bc.detach().cpu()))
            history["rel_l2"].append(eval_data["rel_l2"])
            history["rel_linf"].append(eval_data["rel_linf"])
            history["max_abs_error"].append(eval_data["max_abs_error"])
            history["lr"].append(float(current_lr))

            print(
                f"epoch={epoch:5d} | "
                f"lr={current_lr:.2e} | "
                f"total={loss.item():.4e} | "
                f"obs={loss_obs.item():.4e} | "
                f"pde={loss_pde.item():.4e} | "
                f"comp={loss_comp.item():.4e} | "
                f"bc={loss_bc.item():.4e} | "
                f"relL2={eval_data['rel_l2']:.4e} | "
                f"relLinf={eval_data['rel_linf']:.4e}"
            )

    # Changed data save name
    history_path = out_dir / "history_mlp_pinn_plap_obstacle.npz"

    np.savez(
        history_path,
        total=np.array(history["total"]),
        obs=np.array(history["obs"]),
        pde=np.array(history["pde"]),
        comp=np.array(history["comp"]),
        bc=np.array(history["bc"]),
        rel_l2=np.array(history["rel_l2"]),
        rel_linf=np.array(history["rel_linf"]),
        max_abs_error=np.array(history["max_abs_error"]),
        lr=np.array(history["lr"]),
    )

    # Changed model save name
    model_path = out_dir / "mlp_pinn_plap_obstacle.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "model_type": "MLP-PINN",
            "problem": "p_laplacian_obstacle",
            "p": cfg.p,
            "domain": "[0,2]^2",
            "obstacle": "psi=1 for 0.5 <= x <= 1.5, otherwise 0",
            "loss": "obstacle + PDE inequality + complementarity + boundary loss",
        },
        model_path,
    )

    final = evaluate_on_square(model, cfg)

    print("\nTraining complete.")
    print(f"Final relative L2 error:   {final['rel_l2']:.6e}")
    print(f"Final relative Linf error: {final['rel_linf']:.6e}")
    print(f"Final max absolute error:  {final['max_abs_error']:.6e}")
    print(f"Saved history to: {history_path}")
    print(f"Saved model to:   {model_path}")

    return model, history


if __name__ == "__main__":
    cfg = Config()
    train(cfg)