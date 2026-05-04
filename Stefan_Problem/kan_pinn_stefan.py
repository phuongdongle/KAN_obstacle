import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 12345

    # Space-time box
    x_max: float = 2.25
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # KAN models
    u_widths: tuple = (3, 48, 48, 48, 1)   # input: (x,y,t)
    s_widths: tuple = (2, 32, 32, 1)       # input: (y,t)
    grid_size: int = 15

    # Training
    epochs: int = 10000
    lr: float = 1e-3
    n_interior: int = 2048
    n_boundary: int = 512
    n_free_boundary: int = 512
    n_initial: int = 512
    print_every: int = 250

    scheduler_step: int = 2000
    scheduler_gamma: float = 0.5

    # Loss weights
    lambda_pde: float = 1.0
    lambda_ic: float = 1000.0
    lambda_bc: float = 1000.0
    lambda_fb_u: float = 100.0
    lambda_fb_stefan: float = 100.0
    lambda_s_ic: float = 100.0

    # Evaluation
    n_eval: int = 64

    out_dir: str = "outputs_kan_stefan_2d"


# ------------------------------------------------------------
# Exact solution and free boundary
# ------------------------------------------------------------

def u_exact_fn(xyt: torch.Tensor) -> torch.Tensor:
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    return torch.exp(1.25 * t - x + 0.5 * y + 0.5) - 1.0


def s_exact_fn(yt: torch.Tensor) -> torch.Tensor:
    y = yt[:, 0]
    t = yt[:, 1]
    return 0.5 * y + 1.25 * t + 0.5


def u0_fn(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    return torch.exp(-x + 0.5 * y + 0.5) - 1.0


def s0_fn(y: torch.Tensor) -> torch.Tensor:
    return 0.5 * y + 0.5


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

def rand_uniform(n, low, high, cfg):
    return low + (high - low) * torch.rand(n, 1, device=cfg.device, dtype=cfg.dtype)


def make_interior_points(cfg: Config, s_model: nn.Module):
    y = rand_uniform(cfg.n_interior, cfg.y_min, cfg.y_max, cfg)
    t = rand_uniform(cfg.n_interior, cfg.t_min, cfg.t_max, cfg)

    yt = torch.cat([y, t], dim=1)

    with torch.no_grad():
        s_val = s_model(yt).clamp(min=1e-4, max=cfg.x_max)

    xi = torch.rand(cfg.n_interior, 1, device=cfg.device, dtype=cfg.dtype)
    x = xi * s_val

    return torch.cat([x, y, t], dim=1)


def make_initial_points(cfg: Config):
    y = rand_uniform(cfg.n_initial, cfg.y_min, cfg.y_max, cfg)
    s0 = s0_fn(y.view(-1)).view(-1, 1)

    xi = torch.rand(cfg.n_initial, 1, device=cfg.device, dtype=cfg.dtype)
    x = xi * s0

    t = torch.zeros_like(x)
    xy = torch.cat([x, y], dim=1)
    xyt = torch.cat([x, y, t], dim=1)

    return xyt, xy


def make_fixed_boundary_points(cfg: Config, s_model: nn.Module):
    n = cfg.n_boundary
    side = torch.randint(0, 3, (n, 1), device=cfg.device)

    t = rand_uniform(n, cfg.t_min, cfg.t_max, cfg)
    xyt = torch.zeros(n, 3, device=cfg.device, dtype=cfg.dtype)

    # x = 0
    mask = side[:, 0] == 0
    y = rand_uniform(int(mask.sum()), cfg.y_min, cfg.y_max, cfg)
    xyt[mask, 0:1] = 0.0
    xyt[mask, 1:2] = y
    xyt[mask, 2:3] = t[mask]

    # y = 0
    mask = side[:, 0] == 1
    tt = t[mask]
    yy = torch.zeros_like(tt)
    yt = torch.cat([yy, tt], dim=1)
    with torch.no_grad():
        s_val = s_model(yt).clamp(min=1e-4, max=cfg.x_max)
    xi = torch.rand_like(tt)
    xx = xi * s_val
    xyt[mask, 0:1] = xx
    xyt[mask, 1:2] = yy
    xyt[mask, 2:3] = tt

    # y = 1
    mask = side[:, 0] == 2
    tt = t[mask]
    yy = torch.ones_like(tt)
    yt = torch.cat([yy, tt], dim=1)
    with torch.no_grad():
        s_val = s_model(yt).clamp(min=1e-4, max=cfg.x_max)
    xi = torch.rand_like(tt)
    xx = xi * s_val
    xyt[mask, 0:1] = xx
    xyt[mask, 1:2] = yy
    xyt[mask, 2:3] = tt

    return xyt


def make_free_boundary_points(cfg: Config):
    y = rand_uniform(cfg.n_free_boundary, cfg.y_min, cfg.y_max, cfg)
    t = rand_uniform(cfg.n_free_boundary, cfg.t_min, cfg.t_max, cfg)
    return torch.cat([y, t], dim=1)


# ------------------------------------------------------------
# KAN architecture
# ------------------------------------------------------------

class PiecewiseLinearBasis(nn.Module):
    def __init__(self, grid_size: int, x_min: float = 0.0, x_max: float = 1.0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

        grid = torch.linspace(x_min, x_max, grid_size)
        self.register_buffer("grid", grid)

        self.h = (x_max - x_min) / (grid_size - 1)

    def forward(self, x):
        x = x.clamp(self.x_min, self.x_max)
        return torch.relu(1.0 - torch.abs(x.unsqueeze(-1) - self.grid) / self.h)


class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size):
        super().__init__()
        self.basis = PiecewiseLinearBasis(grid_size)
        self.coeff = nn.Parameter(0.05 * torch.randn(out_dim, in_dim, grid_size))
        self.scale = nn.Parameter(0.05 * torch.randn(out_dim, in_dim))
        self.shift = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        B = self.basis(x)
        basis_val = torch.einsum("big,oig->boi", B, self.coeff)
        affine_val = self.scale.unsqueeze(0) * x.unsqueeze(1) + self.shift.unsqueeze(0)
        return (basis_val + affine_val).sum(dim=-1) + self.bias.unsqueeze(0)


class KANNet(nn.Module):
    def __init__(self, widths, grid_size):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLayer(din, dout, grid_size)
            for din, dout in zip(widths[:-1], widths[1:])
        ])

    def forward(self, x):
        z = x
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = torch.sigmoid(z)
        return z


class NormalizedKAN(nn.Module):
    def __init__(self, widths, grid_size, in_min, in_max):
        super().__init__()
        self.net = KANNet(widths, grid_size)
        self.register_buffer("in_min", torch.tensor(in_min).view(1, -1))
        self.register_buffer("in_max", torch.tensor(in_max).view(1, -1))

    def forward(self, x):
        z = (x - self.in_min) / (self.in_max - self.in_min)
        return self.net(z)


class HardBCStefanUModel(nn.Module):
    def __init__(self, widths, grid_size, x_max):
        super().__init__()

        self.raw_net = NormalizedKAN(
            widths=widths,
            grid_size=grid_size,
            in_min=[0.0, 0.0, 0.0],
            in_max=[x_max, 1.0, 1.0],
        )

    def boundary_interpolant(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]

        # Boundary data from exact solution
        g_left = torch.exp(1.25 * t + 0.5 * y + 0.5) - 1.0       # x = 0
        g_bottom = torch.exp(1.25 * t - x + 0.5) - 1.0           # y = 0
        g_top = torch.exp(1.25 * t - x + 1.0) - 1.0              # y = 1

        # Corner compatibility values
        c_bottom = torch.exp(1.25 * t + 0.5) - 1.0               # x=0,y=0
        c_top = torch.exp(1.25 * t + 1.0) - 1.0                  # x=0,y=1

        # Transfinite interpolation for x=0, y=0, y=1
        B = (
            g_left
            + (1.0 - y) * (g_bottom - c_bottom)
            + y * (g_top - c_top)
        )

        return B

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]

        B = self.boundary_interpolant(xyt)
        raw = self.raw_net(xyt)

        # Vanishes on x=0, y=0, y=1
        factor = x * y * (1.0 - y)

        return B + factor * raw


class HardICStefanSModel(nn.Module):
    def __init__(self, widths, grid_size):
        super().__init__()

        self.raw_net = NormalizedKAN(
            widths=widths,
            grid_size=grid_size,
            in_min=[0.0, 0.0],
            in_max=[1.0, 1.0],
        )

    def forward(self, yt: torch.Tensor) -> torch.Tensor:
        y = yt[:, 0:1]
        t = yt[:, 1:2]

        s0 = 0.5 * y + 0.5

        # Enforces s(y,0) = s0(y)
        return s0 + t * self.raw_net(yt)


# ------------------------------------------------------------
# Differential operators
# ------------------------------------------------------------

def heat_residual(u_model: nn.Module, xyt: torch.Tensor):
    xyt = xyt.clone().detach().requires_grad_(True)

    u = u_model(xyt)

    grad_u = torch.autograd.grad(
        u,
        xyt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]

    u_xx = torch.autograd.grad(
        u_x,
        xyt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y,
        xyt,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
    )[0][:, 1:2]

    res = u_t - u_xx - u_yy

    return xyt, u, grad_u, res


def stefan_residual(u_model: nn.Module, s_model: nn.Module, yt: torch.Tensor):
    yt = yt.clone().detach().requires_grad_(True)

    s = s_model(yt)

    grad_s = torch.autograd.grad(
        s,
        yt,
        grad_outputs=torch.ones_like(s),
        create_graph=True,
    )[0]

    s_y = grad_s[:, 0:1]
    s_t = grad_s[:, 1:2]

    xyt_fb = torch.cat([s, yt[:, 0:1], yt[:, 1:2]], dim=1)
    xyt_fb = xyt_fb.clone().detach().requires_grad_(True)

    u_fb = u_model(xyt_fb)

    grad_u = torch.autograd.grad(
        u_fb,
        xyt_fb,
        grad_outputs=torch.ones_like(u_fb),
        create_graph=True,
    )[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    # Stefan condition:
    # s_t + u_x - s_y u_y = 0
    res_stefan = s_t + u_x - s_y * u_y

    return s, u_fb, res_stefan


# ------------------------------------------------------------
# Error metrics
# ------------------------------------------------------------

def relative_l2_error(pred, truth):
    return float((torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)).detach().cpu())


def relative_linf_error(pred, truth):
    return float((torch.max(torch.abs(pred - truth)) / torch.max(torch.abs(truth))).detach().cpu())


@torch.no_grad()
def evaluate(cfg: Config, u_model: nn.Module, s_model: nn.Module):
    n = cfg.n_eval
    batch_size = 8192

    # --------------------------------------------------
    # Evaluate free boundary s(y,t)
    # --------------------------------------------------
    y = torch.linspace(0.0, 1.0, n, device=cfg.device, dtype=cfg.dtype)
    t = torch.linspace(0.0, 1.0, n, device=cfg.device, dtype=cfg.dtype)

    Y, T = torch.meshgrid(y, t, indexing="ij")
    yt = torch.stack([Y.reshape(-1), T.reshape(-1)], dim=1)

    S_pred_list = []

    for i in range(0, yt.shape[0], batch_size):
        S_pred_list.append(s_model(yt[i:i + batch_size]))

    S_pred = torch.cat(S_pred_list, dim=0)
    S_true = s_exact_fn(yt).view(-1, 1)

    rel_s_l2 = relative_l2_error(S_pred, S_true)
    rel_s_linf = relative_linf_error(S_pred, S_true)

    # --------------------------------------------------
    # Evaluate u(x,y,t) on mapped domain: x = xi * s_exact(y,t)
    # --------------------------------------------------
    xi = torch.linspace(0.0, 1.0, n, device=cfg.device, dtype=cfg.dtype)
    XI, YY, TT = torch.meshgrid(xi, y, t, indexing="ij")

    yt_full = torch.stack(
        [YY.reshape(-1), TT.reshape(-1)],
        dim=1,
    )

    S_true_full = s_exact_fn(yt_full).view(-1, 1)
    X = XI.reshape(-1, 1) * S_true_full

    xyt = torch.cat([X, yt_full], dim=1)

    U_pred_list = []
    U_true_list = []

    for i in range(0, xyt.shape[0], batch_size):
        xyt_batch = xyt[i:i + batch_size]
        U_pred_list.append(u_model(xyt_batch))
        U_true_list.append(u_exact_fn(xyt_batch).view(-1, 1))

    U_pred = torch.cat(U_pred_list, dim=0)
    U_true = torch.cat(U_true_list, dim=0)

    rel_u_l2 = relative_l2_error(U_pred, U_true)
    rel_u_linf = relative_linf_error(U_pred, U_true)

    return rel_u_l2, rel_u_linf, rel_s_l2, rel_s_linf



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

    u_model = HardBCStefanUModel(
        widths=cfg.u_widths,
        grid_size=cfg.grid_size,
        x_max=cfg.x_max,
        ).to(cfg.device, cfg.dtype)

    s_model = HardICStefanSModel(
        widths=cfg.s_widths,
        grid_size=cfg.grid_size,
        ).to(cfg.device, cfg.dtype)

    params = list(u_model.parameters()) + list(s_model.parameters())

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step,
        gamma=cfg.scheduler_gamma,
    )

    history = {
        "total": [],
        "pde": [],
        "ic": [],
        "bc": [],
        "fb_u": [],
        "fb_stefan": [],
        "s_ic": [],
        "rel_u_l2": [],
        "rel_u_linf": [],
        "rel_s_l2": [],
        "rel_s_linf": [],
        "lr": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()

        # PDE interior loss
        xyt_int = make_interior_points(cfg, s_model)
        _, _, _, res_heat = heat_residual(u_model, xyt_int)
        loss_pde = torch.mean(res_heat**2)

        # Initial condition for u
        xyt0, xy0 = make_initial_points(cfg)
        u0_pred = u_model(xyt0)
        u0_true = u0_fn(xy0).view(-1, 1)
        loss_ic = torch.mean((u0_pred - u0_true) ** 2)

        # Fixed boundary conditions
        #xyt_bc = make_fixed_boundary_points(cfg, s_model)
        #ub_pred = u_model(xyt_bc)
        #ub_true = u_exact_fn(xyt_bc).view(-1, 1)
        #loss_bc = torch.mean((ub_pred - ub_true) ** 2)
        loss_bc = torch.tensor(0.0, device=cfg.device, dtype=cfg.dtype)

        
        # Free-boundary conditions
        yt_fb = make_free_boundary_points(cfg)
        s_pred, u_fb, res_stefan = stefan_residual(u_model, s_model, yt_fb)

        loss_fb_u = torch.mean(u_fb**2)
        loss_fb_stefan = torch.mean(res_stefan**2)

        # Initial condition for s
        y0 = rand_uniform(cfg.n_initial, 0.0, 1.0, cfg)
        t0 = torch.zeros_like(y0)
        yt0 = torch.cat([y0, t0], dim=1)

        #s0_pred = s_model(yt0)
        #s0_true = s0_fn(y0.view(-1)).view(-1, 1)
        #loss_s_ic = torch.mean((s0_pred - s0_true) ** 2)
        loss_s_ic = torch.tensor(0.0, device=cfg.device, dtype=cfg.dtype)
        

        loss = (
            cfg.lambda_pde * loss_pde
            + cfg.lambda_ic * loss_ic
            #+ cfg.lambda_bc * loss_bc
            + cfg.lambda_fb_u * loss_fb_u
            + cfg.lambda_fb_stefan * loss_fb_stefan
            #+ cfg.lambda_s_ic * loss_s_ic
        )

        loss.backward(retain_graph = True)
        optimizer.step()
        scheduler.step()

        if epoch % cfg.print_every == 0 or epoch == 1:
            rel_u_l2, rel_u_linf, rel_s_l2, rel_s_linf = evaluate(cfg, u_model, s_model)
            current_lr = optimizer.param_groups[0]["lr"]

            history["total"].append(float(loss.detach().cpu()))
            history["pde"].append(float(loss_pde.detach().cpu()))
            history["ic"].append(float(loss_ic.detach().cpu()))
            history["bc"].append(float(loss_bc.detach().cpu()))
            history["fb_u"].append(float(loss_fb_u.detach().cpu()))
            history["fb_stefan"].append(float(loss_fb_stefan.detach().cpu()))
            history["s_ic"].append(float(loss_s_ic.detach().cpu()))
            history["rel_u_l2"].append(rel_u_l2)
            history["rel_u_linf"].append(rel_u_linf)
            history["rel_s_l2"].append(rel_s_l2)
            history["rel_s_linf"].append(rel_s_linf)
            history["lr"].append(float(current_lr))

            print(
                f"epoch={epoch:5d} | lr={current_lr:.2e} | "
                f"total={loss.item():.4e} | "
                f"pde={loss_pde.item():.4e} | "
                f"ic={loss_ic.item():.4e} | "
                #f"bc={loss_bc.item():.4e} | "
                f"fb_u={loss_fb_u.item():.4e} | "
                f"fb_stefan={loss_fb_stefan.item():.4e} | "
                #f"s_ic={loss_s_ic.item():.4e} | "
                f"relU_L2={rel_u_l2:.4e} | "
                f"relU_Linf={rel_u_linf:.4e} | "
                f"relS_L2={rel_s_l2:.4e} | "
                f"relS_Linf={rel_s_linf:.4e}"
            )

    np.savez(
        out_dir / "history.npz",
        **{k: np.array(v) for k, v in history.items()},
    )

    torch.save(
        {
            "u_model_state_dict": u_model.state_dict(),
            "s_model_state_dict": s_model.state_dict(),
            "config": cfg.__dict__,
            "problem": "direct_2d_one_phase_stefan",
            "exact_u": "exp(5t/4 - x + y/2 + 1/2) - 1",
            "exact_s": "y/2 + 5t/4 + 1/2",
        },
        out_dir / "kan_pinn_stefan_2d.pt",
    )

    rel_u_l2, rel_u_linf, rel_s_l2, rel_s_linf = evaluate(cfg, u_model, s_model)

    print("\nTraining complete.")
    print(f"Final relative U L2 error:     {rel_u_l2:.6e}")
    print(f"Final relative U Linf error:   {rel_u_linf:.6e}")
    print(f"Final relative S L2 error:     {rel_s_l2:.6e}")
    print(f"Final relative S Linf error:   {rel_s_linf:.6e}")
    print(f"Saved history to: {out_dir / 'history.npz'}")
    print(f"Saved model to:   {out_dir / 'kan_pinn_stefan_2d.pt'}")

    return u_model, s_model, history


if __name__ == "__main__":
    cfg = Config()
    train(cfg)