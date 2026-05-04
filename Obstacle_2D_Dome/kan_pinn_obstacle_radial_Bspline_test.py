import math
import csv
from dataclasses import dataclass, replace
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
# Benchmark KAN spline grid size and model size
# Obstacle / free-boundary benchmark on Ω=[0,1]^2
# PDE: -Δu = 0
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

    # Loss weights
    lambda_obs: float = 20.0
    lambda_pde: float = 1.0
    lambda_comp: float = 1.0
    lambda_bc: float = 1e4

    # Evaluation
    n_eval_2d: int = 256

    # Output
    out_dir: str = "outputs_kan_basis_benchmark"


# ============================================================
# Problem definition
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
# KAN model
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


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int):
        super().__init__()

        self.basis = PiecewiseLinearBasis(grid_size=grid_size)

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
        B = self.basis(x)
        spline_val = torch.einsum("big,oig->boi", B, self.coeff)

        affine_val = (
            self.scale.unsqueeze(0) * x.unsqueeze(1)
            + self.shift.unsqueeze(0)
        )

        edge_val = spline_val + affine_val

        return edge_val.sum(dim=-1) + self.bias.unsqueeze(0)


class KANNet(nn.Module):
    def __init__(self, widths, grid_size: int):
        super().__init__()

        self.layers = nn.ModuleList([
            KANLayer(din, dout, grid_size=grid_size)
            for din, dout in zip(widths[:-1], widths[1:])
        ])

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

        self.kan = KANNet(
            widths=widths,
            grid_size=grid_size,
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
# Operators and diagnostics
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

    rel_l2 = relative_l2_error(
        U_pred.reshape(-1, 1),
        U_true.reshape(-1, 1),
    )

    rel_linf = relative_linf_error(
        U_pred.reshape(-1, 1),
        U_true.reshape(-1, 1),
    )

    return {
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# Optional closed-form parameter count for this KANLayer:
# For each layer: out_dim * in_dim * grid_size     coeff
#               + out_dim * in_dim                 scale
#               + out_dim * in_dim                 shift
#               + out_dim                          bias
# So layer params = out_dim * in_dim * (grid_size + 2) + out_dim.
def count_kan_parameters_from_widths(widths, grid_size: int) -> int:
    total = 0

    for din, dout in zip(widths[:-1], widths[1:]):
        total += dout * din * (grid_size + 2) + dout

    return total


# ============================================================
# Training for one benchmark configuration
# ============================================================

def train_one_configuration(cfg: Config):
    set_seed(cfg.seed)

    model = SquareObstacleKANPINN(
        widths=cfg.widths,
        grid_size=cfg.grid_size,
    ).to(cfg.device, cfg.dtype)

    n_params = count_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler,
        gamma=cfg.scheduler_gamma,
    )

    print("\n" + "=" * 80)
    print(f"Training KAN widths={cfg.widths}, grid_size={cfg.grid_size}")
    print(f"Trainable parameters: {n_params}")
    print("=" * 80)

    history_total = []
    history_rel_l2 = []
    history_rel_linf = []

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

            history_total.append(float(loss.detach().cpu()))
            history_rel_l2.append(eval_data["rel_l2"])
            history_rel_linf.append(eval_data["rel_linf"])

            print(
                f"epoch={epoch:5d} | "
                f"loss={loss.item():.4e} | "
                f"relL2={eval_data['rel_l2']:.4e} | "
                f"relLinf={eval_data['rel_linf']:.4e}"
            )

    final = evaluate_on_square(model, cfg)

    result = {
        "widths": cfg.widths,
        "grid_size": cfg.grid_size,
        "params": n_params,
        "final_rel_l2": final["rel_l2"],
        "final_rel_linf": final["rel_linf"],
        "history_total": np.array(history_total),
        "history_rel_l2": np.array(history_rel_l2),
        "history_rel_linf": np.array(history_rel_linf),
    }

    print("\nTraining complete for this configuration.")
    print(f"Final Relative L2 error:   {final['rel_l2']:.6e}")
    print(f"Final Relative Linf error: {final['rel_linf']:.6e}")

    return result


# ============================================================
# Benchmark driver
# ============================================================

def save_results_csv(results, csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "widths",
            "grid_size",
            "trainable_parameters",
            "final_relative_l2_error",
            "final_relative_linf_error",
        ])

        for r in results:
            writer.writerow([
                str(r["widths"]),
                r["grid_size"],
                r["params"],
                f"{r['final_rel_l2']:.16e}",
                f"{r['final_rel_linf']:.16e}",
            ])


def save_results_npz(results, npz_path: Path):
    widths = np.array([str(r["widths"]) for r in results])
    grid_size = np.array([r["grid_size"] for r in results])
    params = np.array([r["params"] for r in results])
    final_rel_l2 = np.array([r["final_rel_l2"] for r in results])
    final_rel_linf = np.array([r["final_rel_linf"] for r in results])

    np.savez(
        npz_path,
        widths=widths,
        grid_size=grid_size,
        params=params,
        final_rel_l2=final_rel_l2,
        final_rel_linf=final_rel_linf,
    )


def plot_benchmark(results, out_dir: Path):
    plt.figure(figsize=(8, 5.5))

    unique_widths = []
    for r in results:
        if r["widths"] not in unique_widths:
            unique_widths.append(r["widths"])

    for widths in unique_widths:
        subset = [r for r in results if r["widths"] == widths]
        subset = sorted(subset, key=lambda z: z["grid_size"])

        grid_sizes = [r["grid_size"] for r in subset]
        rel_l2 = [r["final_rel_l2"] for r in subset]

        # Parameter count depends on grid_size for KAN.
        # To keep your requested legend format, list the parameter counts
        # corresponding to grid_size = [5, 10, 15, 20].
        params_by_grid = [r["params"] for r in subset]
        params_text = "/".join(str(p) for p in params_by_grid)

        label = f"KAN {widths}: {params_text} params"

        plt.plot(
            grid_sizes,
            rel_l2,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=label,
        )

    plt.yscale("log")
    plt.xticks([5, 10, 15, 20])
    plt.xlabel("Grid size")
    plt.ylabel("Final Relative $L^2$ Error")
    plt.title("KAN basis-size benchmark")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "kan_basis_benchmark_rel_l2.png", dpi=300)
    plt.close()


def plot_benchmark_short_legend(results, out_dir: Path):
    """
    Alternative plot with exactly the legend format:
        KAN: [number] params

    Since KAN parameter count changes with grid_size, this version uses
    the parameter count at grid_size = 20 for each width tuple.
    """

    plt.figure(figsize=(8, 6))

    unique_widths = []
    for r in results:
        if r["widths"] not in unique_widths:
            unique_widths.append(r["widths"])

    for widths in unique_widths:
        subset = [r for r in results if r["widths"] == widths]
        subset = sorted(subset, key=lambda z: z["grid_size"])

        grid_sizes = [r["grid_size"] for r in subset]
        rel_l2 = [r["final_rel_l2"] for r in subset]

        params_at_largest_grid = subset[-1]["params"]
        label = f"KAN: {params_at_largest_grid} params"

        plt.plot(
            grid_sizes,
            rel_l2,
            marker="o-",
            linewidth=3.5,
            markersize=9,
            label=label,
        )

    plt.yscale("log", base = 10)
    plt.xticks([5, 10, 15, 20])
    plt.xlabel("B spline basis")
    plt.ylabel("Relative $L^2$ Error")
    plt.title("Obstacle Problem: Error and Bspline Basis")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(out_dir / "kan_basis_benchmark_rel_l2_short_legend.png", dpi=300)
    plt.close()


def run_benchmark():
    base_cfg = Config()

    out_dir = Path(base_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_sizes = [5, 10, 15, 20]

    width_list = [
        (2, 8, 8, 8, 1),
        (2, 16, 16, 16, 1),
        (2, 32, 32, 32, 1),
        (2, 48, 48, 48, 1),
    ]

    print("Using device:", base_cfg.device)

    if base_cfg.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print(f"Contact equation check value: {_check_contact_eq():.8f}")

    results = []

    for widths in width_list:
        for grid_size in grid_sizes:
            cfg = replace(
                base_cfg,
                widths=widths,
                grid_size=grid_size,
                out_dir=str(out_dir),
            )

            result = train_one_configuration(cfg)
            results.append(result)

            # Save intermediate results after every run.
            save_results_npz(
                results,
                out_dir / "kan_basis_benchmark_results.npz",
            )

    plot_benchmark(results, out_dir)

    print("\nBenchmark complete.")
    print(f"Saved NPZ results to: {out_dir / 'kan_basis_benchmark_results.npz'}")
    print(f"Saved short-legend plot to: {out_dir / 'kan_basis_benchmark_rel_l2_short_legend.png'}")


if __name__ == "__main__":
    run_benchmark()