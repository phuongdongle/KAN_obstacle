import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "lines.linewidth": 3.0,
    "grid.alpha": 0.7,
})



# ============================================================
# Compare PINN, ResNet, and KAN for the DOME obstacle problem
# Domain: Ω = [0,1]^2
#
# PDE inequality:
#   -Δu >= 0
#
# Obstacle:
#   radial dome centered at (1/2, 1/2)
#
# Exact solution:
#   contact region: r <= r_*
#   non-contact region: radial logarithmic solution
# ============================================================


def set_seed(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 12345

    # Training
    epochs: int = 10000
    lr: float = 1e-3
    n_interior: int = 2048
    n_boundary: int = 1024
    print_every: int = 250

    # Scheduler
    scheduler_step: int = 2000
    scheduler_gamma: float = 0.2

    # Loss weights
    lambda_obs: float = 20.0
    lambda_pde: float = 1.0
    lambda_comp: float = 1.0
    lambda_bc: float = 1e4

    # KAN
    grid_size: int = 10
    use_rbf_kan: bool = False

    # ResNet
    n_blocks: int = 4

    # Evaluation
    n_eval_2d: int = 128

    # Output
    out_dir: str = "outputs_dome_method_width_comparison"


# ============================================================
# Dome obstacle problem definition
# ============================================================

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


# ============================================================
# Sampling
# ============================================================

def make_interior_points(cfg: Config) -> torch.Tensor:
    return torch.rand(cfg.n_interior, 2, device=cfg.device, dtype=cfg.dtype)


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


# ============================================================
# Activation helper
# ============================================================

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

    raise ValueError(f"Unknown activation: {name}")


# ============================================================
# PINN / MLP model, soft boundary condition
# ============================================================

class MLPNet(nn.Module):
    def __init__(self, widths, activation: str = "tanh"):
        super().__init__()

        layers = []

        for din, dout in zip(widths[:-2], widths[1:-1]):
            layers.append(nn.Linear(din, dout))
            layers.append(get_activation(activation))

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


class DomePINN(nn.Module):
    def __init__(self, widths, activation: str = "tanh"):
        super().__init__()
        self.mlp = MLPNet(widths=widths, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soft boundary condition only.
        return self.mlp(x)


# ============================================================
# ResNet model, soft boundary condition
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str = "tanh"):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(width, width),
            get_activation(activation),
            nn.Linear(width, width),
        )

        self.activation = get_activation(activation)
        self.initialize()

    def initialize(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        out_dim: int = 1,
        n_blocks: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()

        self.input_layer = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation=activation)
            for _ in range(n_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.activation = get_activation(activation)

        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.input_layer(x))

        for block in self.blocks:
            z = block(z)

        return self.output_layer(z)


class DomeResNetPINN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_blocks: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()

        self.resnet = ResNet(
            in_dim=2,
            hidden_dim=hidden_dim,
            out_dim=1,
            n_blocks=n_blocks,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soft boundary condition only.
        return self.resnet(x)


# ============================================================
# KAN model, hard boundary condition
# ============================================================

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
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int,
        use_rbf: bool = False,
    ):
        super().__init__()

        if use_rbf:
            self.basis = RBFBasis(grid_size=grid_size)
        else:
            self.basis = PiecewiseLinearBasis(grid_size=grid_size)

        self.coeff = nn.Parameter(0.05 * torch.randn(out_dim, in_dim, grid_size))
        self.scale = nn.Parameter(0.05 * torch.randn(out_dim, in_dim))
        self.shift = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = self.basis(x)

        basis_val = torch.einsum("big,oig->boi", B, self.coeff)

        affine_val = (
            self.scale.unsqueeze(0) * x.unsqueeze(1)
            + self.shift.unsqueeze(0)
        )

        edge_val = basis_val + affine_val

        return edge_val.sum(dim=-1) + self.bias.unsqueeze(0)


class KANNet(nn.Module):
    def __init__(self, widths, grid_size: int, use_rbf: bool = False):
        super().__init__()

        self.layers = nn.ModuleList([
            KANLayer(
                in_dim=din,
                out_dim=dout,
                grid_size=grid_size,
                use_rbf=use_rbf,
            )
            for din, dout in zip(widths[:-1], widths[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x

        for i, layer in enumerate(self.layers):
            z = layer(z)

            if i < len(self.layers) - 1:
                z = torch.sigmoid(z)

        return z


class DomeKANPINN(nn.Module):
    def __init__(
        self,
        widths,
        grid_size: int,
        use_rbf: bool = False,
    ):
        super().__init__()

        self.kan = KANNet(
            widths=widths,
            grid_size=grid_size,
            use_rbf=use_rbf,
        )

    def boundary_factor(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        return x1 * (1.0 - x1) * x2 * (1.0 - x2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = g_fn(x).view(-1, 1)
        raw = self.kan(x)

        # Hard boundary matching exact boundary data.
        return g + self.boundary_factor(x) * raw


# ============================================================
# Laplacian operator and diagnostics
# ============================================================

def laplacian(model: nn.Module, x: torch.Tensor):
    x = x.clone().detach().requires_grad_(True)

    u = model(x)

    grad_u = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        grad_u[:, 0:1],
        x,
        grad_outputs=torch.ones_like(grad_u[:, 0:1]),
        create_graph=True,
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        grad_u[:, 1:2],
        x,
        grad_outputs=torch.ones_like(grad_u[:, 1:2]),
        create_graph=True,
    )[0][:, 1:2]

    return x, u, grad_u, u_xx + u_yy


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

    x1 = torch.linspace(0.0, 1.0, n, device=cfg.device, dtype=cfg.dtype)
    x2 = torch.linspace(0.0, 1.0, n, device=cfg.device, dtype=cfg.dtype)

    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    pts = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

    U_pred = model(pts).reshape(n, n)
    U_true = u_true_fn(pts).reshape(n, n)
    PSI = psi_fn(pts).reshape(n, n)
    ERR = torch.abs(U_pred - U_true)

    rel_l2 = relative_l2_error(U_pred.reshape(-1, 1), U_true.reshape(-1, 1))
    rel_linf = relative_linf_error(U_pred.reshape(-1, 1), U_true.reshape(-1, 1))

    fb_idx = torch.argmin(torch.abs((U_pred - PSI).reshape(-1)))
    fb_point = pts[fb_idx]
    fb_radius = float(r_center(fb_point.view(1, 2)).item())

    return {
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
        "max_abs_error": float(torch.max(ERR).detach().cpu()),
        "fb_radius": fb_radius,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Build model
# ============================================================

def build_model(method: str, width: int, cfg: Config) -> nn.Module:
    widths = (2, width, width, width, 1)

    if method == "PINN":
        model = DomePINN(
            widths=widths,
            activation="tanh",
        )

    elif method == "ResNet":
        model = DomeResNetPINN(
            hidden_dim=width,
            n_blocks=cfg.n_blocks,
            activation="tanh",
        )

    elif method == "KAN":
        model = DomeKANPINN(
            widths=widths,
            grid_size=cfg.grid_size,
            use_rbf=cfg.use_rbf_kan,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return model.to(cfg.device, cfg.dtype)


# ============================================================
# Train one configuration
# ============================================================

def train_one(method: str, width: int, cfg: Config):
    set_seed(cfg.seed)

    model = build_model(method, width, cfg)
    n_params = count_trainable_parameters(model)

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

    print("\n" + "=" * 90)
    print(f"Training method={method}, width={width}, params={n_params}")
    print("=" * 90)

    history = {
        "total": [],
        "obs": [],
        "pde": [],
        "comp": [],
        "bc": [],
        "rel_l2": [],
        "rel_linf": [],
        "max_abs_error": [],
        "fb_radius": [],
        "lr": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()

        x_interior = make_interior_points(cfg)

        x_req, u, _, lap_u = laplacian(model, x_interior)

        psi = psi_fn(x_req).view(-1, 1)
        f = f_fn(x_req).view(-1, 1)

        Lu = -lap_u

        # Obstacle constraint: u >= psi
        obs_violation = torch.relu(psi - u)

        # PDE inequality: -Δu - f >= 0
        pde_violation = torch.relu(f - Lu)

        # Complementarity: (u - psi)(Lu - f) = 0
        comp_term = (u - psi) * (Lu - f)

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
            history["fb_radius"].append(eval_data["fb_radius"])
            history["lr"].append(float(current_lr))

            print(
                f"epoch={epoch:5d} | "
                f"lr={current_lr:.2e} | "
                f"loss={loss.item():.4e} | "
                f"obs={loss_obs.item():.4e} | "
                f"pde={loss_pde.item():.4e} | "
                f"comp={loss_comp.item():.4e} | "
                f"bc={loss_bc.item():.4e} | "
                f"relL2={eval_data['rel_l2']:.4e} | "
                f"relLinf={eval_data['rel_linf']:.4e} | "
                f"fb_r={eval_data['fb_radius']:.6f}"
            )

    final = evaluate_on_square(model, cfg)

    result = {
        "method": method,
        "width": width,
        "params": n_params,
        "final_rel_l2": final["rel_l2"],
        "final_rel_linf": final["rel_linf"],
        "final_max_abs_error": final["max_abs_error"],
        "final_fb_radius": final["fb_radius"],
    }

    print("\nTraining complete for this configuration.")
    print(f"Method: {method}")
    print(f"Width: {width}")
    print(f"Trainable parameters: {n_params}")
    print(f"Final relative L2 error:   {final['rel_l2']:.6e}")
    print(f"Final relative Linf error: {final['rel_linf']:.6e}")
    print(f"Final max absolute error:  {final['max_abs_error']:.6e}")
    print(f"Approximate free-boundary radius: {final['fb_radius']:.6f}")

    return model, history, result


# ============================================================
# Save utilities
# ============================================================

def save_history_npz(history: dict, save_path: Path):
    np.savez(
        save_path,
        total=np.array(history["total"]),
        obs=np.array(history["obs"]),
        pde=np.array(history["pde"]),
        comp=np.array(history["comp"]),
        bc=np.array(history["bc"]),
        rel_l2=np.array(history["rel_l2"]),
        rel_linf=np.array(history["rel_linf"]),
        max_abs_error=np.array(history["max_abs_error"]),
        fb_radius=np.array(history["fb_radius"]),
        lr=np.array(history["lr"]),
    )


def save_results_csv(results, csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "method",
            "width",
            "trainable_parameters",
            "final_relative_l2_error",
            "final_relative_linf_error",
            "final_max_absolute_error",
            "final_free_boundary_radius",
        ])

        for r in results:
            writer.writerow([
                r["method"],
                r["width"],
                r["params"],
                f"{r['final_rel_l2']:.16e}",
                f"{r['final_rel_linf']:.16e}",
                f"{r['final_max_abs_error']:.16e}",
                f"{r['final_fb_radius']:.16e}",
            ])


def save_results_npz(results, npz_path: Path):
    np.savez(
        npz_path,
        method=np.array([r["method"] for r in results]),
        width=np.array([r["width"] for r in results]),
        params=np.array([r["params"] for r in results]),
        final_rel_l2=np.array([r["final_rel_l2"] for r in results]),
        final_rel_linf=np.array([r["final_rel_linf"] for r in results]),
        final_max_abs_error=np.array([r["final_max_abs_error"] for r in results]),
        final_fb_radius=np.array([r["final_fb_radius"] for r in results]),
    )


# ============================================================
# Plot comparison
# ============================================================

def plot_method_width_comparison(results, out_dir: Path):
    plt.figure(figsize=(8, 5.5))

    methods = ["PINN", "ResNet", "KAN"]

    for method in methods:
        subset = [r for r in results if r["method"] == method]
        subset = sorted(subset, key=lambda z: z["width"])

        widths = [r["width"] for r in subset]
        rel_l2 = [r["final_rel_l2"] for r in subset]

        plt.plot(
            widths,
            rel_l2,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=method,
        )

    plt.yscale("log")
    plt.xticks([8, 16, 32, 48])
    plt.xlabel("Width")
    plt.ylabel(r"Final Relative $L^2$ Error")
    plt.title("Method comparison for the dome obstacle problem")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(
        out_dir / "dome_method_width_comparison_rel_l2.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_method_width_comparison_with_params(results, out_dir: Path):
    plt.figure(figsize=(8, 6))

    methods = ["PINN", "ResNet", "KAN"]

    for method in methods:
        subset = [r for r in results if r["method"] == method]
        subset = sorted(subset, key=lambda z: z["width"])

        widths = [r["width"] for r in subset]
        rel_l2 = [r["final_rel_l2"] for r in subset]
        params_text = "/".join(str(r["params"]) for r in subset)
        params_val = [r["params"] for r in subset]

        label = f"{method}"

        plt.plot(
            params_val,
            rel_l2,
            marker="d",
            linestyle = "-.",
            linewidth=3,
            markersize=9,
            label=label,
        )

    plt.yscale("log", base = 10)
    plt.xscale("log", base = 2)
    #all_params = sorted(set(r["params"] for r in results))
    #plt.xticks(all_params, [str(p) for p in all_params], rotation=30, ha="right")

    
    plt.xlabel("Total Trainable Parameters")
    plt.ylabel(r"Relative $L^2$ Error")
    plt.title("Obstacle Problem: Error vs. Parameters")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        fontsize=15,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(
        out_dir / "dome_method_width_comparison_rel_l2_with_params.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# Benchmark driver
# ============================================================

def run_benchmark():
    cfg = Config()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", cfg.device)

    if cfg.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print(f"Contact equation check value: {_check_contact_eq():.8f}")

    widths = [16, 32, 48]
    widths_kan = [10, 15, 20]
    methods = ["PINN", "ResNet", "KAN"]

    results = []

    for method in methods:
        current_widths = widths_kan if method == "KAN" else widths
        for width in current_widths:
            run_dir = out_dir / method / f"width_{width}"
            run_dir.mkdir(parents=True, exist_ok=True)

            model, history, result = train_one(
                method=method,
                width=width,
                cfg=cfg,
            )

            save_history_npz(
                history,
                run_dir / f"history_{method.lower()}_width_{width}.npz",
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "method": method,
                    "width": width,
                    "trainable_parameters": result["params"],
                    "problem": "dome_obstacle_problem",
                    "domain": "[0,1]^2",
                    "r_star": r_star,
                    "r_c": r_c,
                    "R_outer": R_outer,
                    "loss": "obstacle + PDE inequality + complementarity + boundary loss",
                    "boundary_enforcement": (
                        "hard ansatz for KAN; soft penalty for PINN and ResNet"
                    ),
                },
                run_dir / f"{method.lower()}_width_{width}.pt",
            )

            results.append(result)

            # Save intermediate benchmark results after each run.

            save_results_npz(
                results,
                out_dir / "dome_method_width_comparison_results.npz",
            )

    #plot_method_width_comparison(results, out_dir)
    plot_method_width_comparison_with_params(results, out_dir)

    print("\nBenchmark complete.")
    print(f"Saved NPZ to: {out_dir / 'dome_method_width_comparison_results.npz'}")
    print(f"Saved parameter plot to: {out_dir / 'dome_method_width_comparison_rel_l2_with_params.png'}")


if __name__ == "__main__":
    run_benchmark()