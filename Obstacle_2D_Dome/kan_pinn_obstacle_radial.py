import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# KAN-PINN for an obstacle / free-boundary benchmark on Ω=[0,1]^2
# PDE:    -Δu = 0  in Ω
# Obstacle: radial dome centered at (1/2,1/2)
# Boundary data: g = u_exact on ∂Ω (hard-enforced)
# ============================================================


def set_seed(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 12345

    # Model
    widths: tuple = (2, 48, 48, 48, 1)
    grid_size: int = 12

    # Training
    epochs: int = 10_000
    lr: float = 1e-3
    n_interior: int = 2048
    n_boundary: int = 1024
    print_every: int = 250
    scheduler: int = 2000
    scheduler_gamma: float = 0.2

    # lambda weight values. 
    lambda_obs: float = 20.0
    lambda_pde: float = 1.0
    lambda_comp: float = 1.0
    lambda_bc: float = 1e4

    # Output / plots
    out_dir: str = "outputs_kan_obstacle_dome_square"
    n_eval_2d: int = 256


# ------------------------------------------------------------------
# Free-boundary / obstacle benchmark on Ω = [0,1]^2
# PDE:  -Δu = 0  in Ω   ⇒  f = 0
# Center: (1/2, 1/2)
# Inner radius: r_c = 1/2
# Outer radius (used in the closed form): R = 1
# Contact radius: r_* ≈ 0.260197
# ------------------------------------------------------------------
r_c = 0.5
R_outer = 1.0
r_star = 0.260197


def _check_contact_eq() -> float:
    return (r_star**2 / r_c**2) * (1.0 + 2.0 * math.log(R_outer / r_star))


def r_center(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, 0] - 0.5
    dy = x[:, 1] - 0.5
    return torch.sqrt(torch.clamp(dx * dx + dy * dy, min=0.0))


def psi_fn(x: torch.Tensor) -> torch.Tensor:
    r = r_center(x)
    inside = r <= r_c
    val_inside = 1.0 - (r * r) / (r_c * r_c)
    return torch.where(inside, val_inside, torch.zeros_like(r))


def u_true_fn(x: torch.Tensor) -> torch.Tensor:
    r = r_center(x)
    inside = r <= r_star

    u_inside = 1.0 - (r * r) / (r_c * r_c)

    r_safe = torch.clamp(r, min=1e-12)
    coef = (r_star * r_star) / (r_c * r_c)
    u_outside = 1.0 - coef * (1.0 + 2.0 * torch.log(r_safe / r_star))

    return torch.where(inside, u_inside, u_outside)


def f_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


def g_fn(x: torch.Tensor) -> torch.Tensor:
    return u_true_fn(x)


def make_boundary_points(cfg: Config) -> torch.Tensor:
    n = cfg.n_boundary
    device, dtype = cfg.device, cfg.dtype

    s = torch.rand(n, 1, device=device, dtype=dtype)
    side = torch.randint(0, 4, (n, 1), device=device)

    x = torch.zeros(n, 2, device=device, dtype=dtype)

    # x1 = 0
    mask = side[:, 0] == 0
    x[mask, 0] = 0.0
    x[mask, 1] = s[mask, 0]

    # x1 = 1
    mask = side[:, 0] == 1
    x[mask, 0] = 1.0
    x[mask, 1] = s[mask, 0]

    # x2 = 0
    mask = side[:, 0] == 2
    x[mask, 0] = s[mask, 0]
    x[mask, 1] = 0.0

    # x2 = 1
    mask = side[:, 0] == 3
    x[mask, 0] = s[mask, 0]
    x[mask, 1] = 1.0

    return x


# ------------------------------------------------------------
# Spline-KAN building blocks.
# ------------------------------------------------------------
class PiecewiseLinearBasis(nn.Module):
    def __init__(self, grid_size: int, x_min: float = 0.0, x_max: float = 1.0):
        super().__init__()
        assert grid_size >= 2
        self.x_min = x_min
        self.x_max = x_max
        grid = torch.linspace(x_min, x_max, grid_size)
        self.register_buffer("grid", grid)
        self.h = (x_max - x_min) / (grid_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.x_min, self.x_max)
        return torch.relu(1.0 - torch.abs(x.unsqueeze(-1) - self.grid) / self.h)


class RBFBasis(nn.Module):
    def __init__(self, grid_size: int, x_min: float = 0.0, x_max: float = 1.0):
        super().__init__()
        assert grid_size >= 2
        self.x_min = x_min
        self.x_max = x_max

        centers = torch.linspace(x_min, x_max, grid_size)
        self.register_buffer("centers", centers)

        h = (x_max - x_min) / (grid_size - 1)
        self.gamma = 1.0 / (h ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.x_min, self.x_max)
        diff = x.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff**2)



class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int):
        super().__init__()
        self.basis = PiecewiseLinearBasis(grid_size=grid_size)
        self.coeff = nn.Parameter(0.05 * torch.randn(out_dim, in_dim, grid_size))
        self.scale = nn.Parameter(0.05 * torch.randn(out_dim, in_dim))
        self.shift = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = self.basis(x)  # [B, in_dim, G]
        spline_val = torch.einsum("big,oig->boi", B, self.coeff)
        affine_val = self.scale.unsqueeze(0) * x.unsqueeze(1) + self.shift.unsqueeze(0)
        edge_val = spline_val + affine_val
        return edge_val.sum(dim=-1) + self.bias.unsqueeze(0)


class RBFKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int):
        super().__init__()

        self.basis = RBFBasis(grid_size=grid_size)

        self.coeff = nn.Parameter(
            0.05 * torch.randn(out_dim, in_dim, grid_size)
        )

        self.scale = nn.Parameter(
            0.05 * torch.randn(out_dim, in_dim)
        )

        self.shift = nn.Parameter(
            torch.zeros(out_dim, in_dim)
        )

        self.bias = nn.Parameter(
            torch.zeros(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = self.basis(x)  # [batch, in_dim, grid_size]

        rbf_val = torch.einsum("big,oig->boi", B, self.coeff)

        affine_val = (
            self.scale.unsqueeze(0) * x.unsqueeze(1)
            + self.shift.unsqueeze(0)
        )

        edge_val = rbf_val + affine_val

        return edge_val.sum(dim=-1) + self.bias.unsqueeze(0)


class KANNet(nn.Module):
    def __init__(self, widths, grid_size: int):
        super().__init__()
        
        self.layers = nn.ModuleList([
            KANLayer(din, dout, grid_size=grid_size)
            for din, dout in zip(widths[:-1], widths[1:])
        ])

        #self.layers = nn.ModuleList([
        #    RBFKANLayer(din, dout, grid_size=grid_size)
        #    for din, dout in zip(widths[:-1], widths[1:])
        #])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = torch.sigmoid(z)
        return z


class SquareObstacleKANPINN(nn.Module):
    def __init__(self, widths, grid_size: int):
        super().__init__()
        self.kan = KANNet(widths=widths, grid_size=grid_size)

    def boundary_factor(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        return x1 * (1.0 - x1) * x2 * (1.0 - x2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = g_fn(x).view(-1, 1)
        raw = self.kan(x)
        return g + self.boundary_factor(x) * raw


# ------------------------------------------------------------
# Differential operators and sampling.
# ------------------------------------------------------------
def laplacian(model: nn.Module, x: torch.Tensor):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(grad_u[:, 0:1], x, grad_outputs=torch.ones_like(grad_u[:, 0:1]), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(grad_u[:, 1:2], x, grad_outputs=torch.ones_like(grad_u[:, 1:2]), create_graph=True)[0][:, 1:2]
    return x, u, grad_u, u_xx + u_yy


def make_interior_points(cfg: Config) -> torch.Tensor:
    return torch.rand(cfg.n_interior, 2, device=cfg.device, dtype=cfg.dtype)


# ------------------------------------------------------------
# Diagnostics and plotting.
# ------------------------------------------------------------
def relative_l2_error(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)).detach().cpu())


def relative_linf_error(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(
        (torch.max(torch.abs(pred - truth)) / torch.max(torch.abs(truth)))
        .detach()
        .cpu()
    )


@torch.no_grad()
def evaluate_on_square(model: nn.Module, cfg: Config):
    n = cfg.n_eval_2d
    x1 = torch.linspace(0.0, 1.0, n, device=cfg.device, dtype=cfg.dtype)
    x2 = torch.linspace(0.0, 1.0, n, device=cfg.device, dtype=cfg.dtype)
    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    pts = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

    U_pred = model(pts).reshape(n, n)
    U_true = u_true_fn(pts).reshape(n, n)
    PSI = psi_fn(pts).reshape(n, n)
    ERR = torch.abs(U_pred - U_true)
    
    rel_linf = relative_linf_error(U_pred.reshape(-1, 1), U_true.reshape(-1, 1))
    rel_l2 = relative_l2_error(U_pred.reshape(-1, 1), U_true.reshape(-1, 1))
    fb_idx = torch.argmin(torch.abs((U_pred - PSI).reshape(-1)))
    fb_point = pts[fb_idx]
    fb_radius = float(r_center(fb_point.view(1, 2)).item())

    return {
        "X": X1.detach().cpu().numpy(),
        "Y": X2.detach().cpu().numpy(),
        "U_pred": U_pred.detach().cpu().numpy(),
        "U_true": U_true.detach().cpu().numpy(),
        "PSI": PSI.detach().cpu().numpy(),
        "ERR": ERR.detach().cpu().numpy(),
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
        "fb_radius": fb_radius,
    }



# ------------------------------------------------------------
# Training.
# ------------------------------------------------------------
def train(cfg: Config):
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", cfg.device)
    if cfg.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Contact equation check value: {_check_contact_eq():.8f}")

    model = SquareObstacleKANPINN(widths=cfg.widths, grid_size=cfg.grid_size).to(cfg.device, cfg.dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=cfg.scheduler, gamma=cfg.scheduler_gamma)
    


    
    history = {"total": [], "obs": [], "pde": [], "comp": [], "bc": [], "rel_l2": [], "rel_linf": []}

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()

        x_interior = make_interior_points(cfg)
        x_req, u, _, lap_u = laplacian(model, x_interior)
        psi = psi_fn(x_req).view(-1, 1)
        f = f_fn(x_req).view(-1, 1)

        Lu = -lap_u
        obs_violation = torch.relu(psi - u)
        pde_violation = torch.relu(f - Lu)
        comp_term = (u - psi) * (Lu - f)

        loss_obs = torch.mean(obs_violation**2)
        loss_pde = torch.mean(pde_violation**2)
        loss_comp = torch.mean(comp_term**2)


        # Boundary loss on random boundary points
        x_boundary = make_boundary_points(cfg)
        u_boundary = model(x_boundary)
        g_boundary = g_fn(x_boundary).view(-1, 1)
        loss_bc = torch.mean((u_boundary - g_boundary)**2)

        loss = (
            cfg.lambda_obs * loss_obs
            + cfg.lambda_pde * loss_pde
            + cfg.lambda_comp * loss_comp
            + cfg.lambda_bc * loss_bc
        )
        loss.backward(retain_graph = True)
        optimizer.step()

        if epoch % cfg.print_every == 0 or epoch == 1:
            eval_data = evaluate_on_square(model, cfg)
            history["total"].append(float(loss.detach().cpu()))
            history["obs"].append(float(loss_obs.detach().cpu()))
            history["pde"].append(float(loss_pde.detach().cpu()))
            history["comp"].append(float(loss_comp.detach().cpu()))
            history["bc"].append(float(loss_bc.detach().cpu()))
            history["rel_l2"].append(eval_data["rel_l2"])
            history["rel_linf"].append(eval_data["rel_linf"])
            print(
                f"epoch={epoch:5d} | total={loss.item():.4e} "
                f"obs={loss_obs.item():.4e} "
                f"pde={loss_pde.item():.4e} "
                f"comp={loss_comp.item():.4e} "
                f"relL2={eval_data['rel_l2']:.4e} "
                f"relLinf={eval_data['rel_linf']:.4e} "
                #f"fb_r≈{eval_data['fb_radius']:.6f}"
            )

    np.savez(
        out_dir / "history.npz",
        total=np.array(history["total"]),
        obs=np.array(history["obs"]),
        pde=np.array(history["pde"]),
        comp=np.array(history["comp"]),
        bc=np.array(history["bc"]),
        rel_l2=np.array(history["rel_l2"]),
        rel_linf=np.array(history["rel_linf"]),
    )

    save_plots(model, cfg, out_dir)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "r_star": r_star,
            "r_c": r_c,
            "R_outer": R_outer,
        },
        out_dir / "kan_pinn_obstacle_dome_square.pt",
    )

    final = evaluate_on_square(model, cfg)
    print("\nTraining complete.")
    print(f"Final relative L2 error: {final['rel_l2']:.6e}")
    print(f"Final relative Linf error: {final['rel_linf']:.6e}")
    print(f"Approximate free boundary radius: r ≈ {final['fb_radius']:.6f}")
    print(f"Artifacts written to: {out_dir.resolve()}")
    return model, history







if __name__ == "__main__":
    cfg = Config()
    train(cfg)
    