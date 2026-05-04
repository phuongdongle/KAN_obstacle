import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker
# ============================================================
# KAN-PINN for an obstacle / free-boundary benchmark on Ω=[0,1]^2
# PDE:    -Δu = 0  in Ω
# Obstacle: radial dome centered at (1/2,1/2)
# Boundary data: g = u_exact on ∂Ω (hard-enforced)
# ============================================================



plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "lines.linewidth": 3.0,
    "lines.markersize": 12,
    "grid.alpha": 0.7,
})



def set_seed(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sci_notation(x, pos):
    if x == 0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exp
    return rf"${coeff:.0f}\times 10^{{{exp}}}$"


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


def save_surface(X, Y, Z, title: str, zlabel: str, out_path: Path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        shade = True,
        cstride = 2, rstride = 2
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    #ax.set_zlabel(zlabel)
    ax.set_title(title)

    cbar = fig.colorbar(surf, shrink=0.7, aspect=16)
    cbar.set_label(zlabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def save_training_curves(history_path: Path, out_dir: Path, print_every: int):
    history = np.load(history_path)

    n_log = len(history["total"])
    epochs_logged = np.arange(n_log) * print_every + 1

    if n_log > 1:
        epochs_logged[1:] = np.arange(1, n_log) * print_every

    plt.figure(figsize=(8, 5))

    plt.plot(epochs_logged, history["total"], "k-", linewidth=3, label="total loss")
    plt.plot(epochs_logged, history["obs"], "b--", linewidth=3, label="obstacle loss")
    plt.plot(epochs_logged, history["pde"], "r--", linewidth=3, label="PDE loss")
    plt.plot(epochs_logged, history["comp"], "g--", linewidth=3, label="complementarity loss")

    #if "bc" in history.files:
    #    plt.plot(epochs_logged, history["bc"], "m--", linewidth=2, label="boundary loss")

    plt.yscale("log")
    plt.xlabel("Number of epochs", fontsize=18)
    plt.ylabel("Loss Value", fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_components_logscale.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs_logged,
        history["rel_l2"],
        "b--",
        linewidth=3,
        label=r"$\|u_{\mathrm{exact}}-u_{\mathrm{KAN}}\|_{L^2}/\|u_{\mathrm{exact}}\|_{L^2}$",
    )

    plt.plot(
        epochs_logged,
        history["rel_linf"],
        "r-.",
        linewidth=3,
        label=r"$\|u_{\mathrm{exact}}-u_{\mathrm{KAN}}\|_{L^\infty}/\|u_{\mathrm{exact}}\|_{L^\infty}$",
    )

    plt.yscale("log", base = 10)
    plt.xlabel("Number of epochs", fontsize=18)
    plt.ylabel("Relative error", fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "relative_errors_logscale.png", dpi=300)
    plt.close()


def save_solution_plots(model: nn.Module, cfg: Config, out_dir: Path):
    data = evaluate_on_square(model, cfg)

    X = data["X"]
    Y = data["Y"]
    U_true = data["U_true"]
    U_pred = data["U_pred"]
    PSI = data["PSI"]
    ERR = data["ERR"]

    save_surface(
        X,
        Y,
        U_true,
        "Exact solution",
        r"$u_{\mathrm{exact}}$",
        out_dir / "u_exact_3d.png",
    )

    save_surface(
        X,
        Y,
        U_pred,
        "KAN approximation",
        r"$u_{\mathrm{KAN}}$",
        out_dir / "u_pred_3d.png",
    )


    save_surface(
    X,
    Y,
    PSI,
    "Obstacle function",
    r"$\psi$",
    out_dir / "obstacle_3d.png",
    )

    R = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    contact_mask = R <= r_star

    plt.figure(figsize=(6, 5))

    im = plt.pcolormesh(
        X,
        Y,
        ERR.T,
        cmap="jet",
        shading="auto",
    )

    #plt.contourf(
    #    X,
    #    Y,
    #    contact_mask.astype(float).T,
    #    levels=[0.5, 1.5],
    #    colors=["white"],
    #    alpha=0.25,
    #)

    #cs = plt.contour(
    #    X,
    #    Y,
    #    R.T,
    #    levels=[r_star],
    #    colors="green",
    #    linewidths=1.2
    #)

    #cs.collections[0].set_label(rf"$r^* = {r_star:.3f}$")

    #plt.legend(loc="upper right", fontsize=15)
    
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Absolute error")
    cbar = plt.colorbar(im, label=r"$|u_{\mathrm{KAN}}-u_{\mathrm{exact}}|$")

    # Force scientific notation 
    sf = mticker.ScalarFormatter(useMathText=True, useOffset = False)
    sf.set_powerlimits((0, 0))        # always show scientific notation
    cbar.locator   = mticker.LinearLocator(numticks=6)  #  evenly spaced ticks
    cbar.formatter = sf
    cbar.update_ticks()

    plt.tight_layout()
    plt.savefig(out_dir / "u_abs_error_2d_contact_region.png", dpi=300)
    plt.close()

    for arr, name, title in [
        (U_true, "u_exact_2d.png", "Exact solution"),
        (U_pred, "u_pred_2d.png", "KAN approximation"),
        (PSI, "obstacle_2d.png", "Obstacle"),
    ]:
        plt.figure(figsize=(6, 5))

        im = plt.pcolormesh(
            X,
            Y,
            arr.T,
            cmap="jet",
            shading="auto",
        )

        plt.gca().set_aspect("equal")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.title(title)
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=300)
        plt.close()



    





    # ------------------------------------------------------------
    # Save 3D sampling points separately
    # green = interior points, red = boundary points
    # ------------------------------------------------------------
    x_int = torch.rand(cfg.n_interior, 2, device=cfg.device, dtype=cfg.dtype)
    x_bnd = make_boundary_points(cfg)

    z_int = psi_fn(x_int).detach().cpu().numpy()
    z_bnd = psi_fn(x_bnd).detach().cpu().numpy()

    x_int_np = x_int.detach().cpu().numpy()
    x_bnd_np = x_bnd.detach().cpu().numpy()

    # Optional: reduce number of displayed points for clarity
    n_show_int = min(1200, cfg.n_interior)
    n_show_bnd = min(800, cfg.n_boundary)

    idx_int = np.random.choice(cfg.n_interior, n_show_int, replace=False)
    idx_bnd = np.random.choice(cfg.n_boundary, n_show_bnd, replace=False)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot obstacle lightly underneath
    ax.plot_surface(
        X,
        Y,
        PSI,
        cmap="gray",
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.35,
        rstride=4,
        cstride=4,
    )

    # Interior points
    ax.scatter(
        x_int_np[idx_int, 0],
        x_int_np[idx_int, 1],
        z_int[idx_int],
        c="green",
        s=10,
        alpha=0.8,
        label=rf"interior points $\Omega$",
    )

    # Boundary points
    ax.scatter(
        x_bnd_np[idx_bnd, 0],
        x_bnd_np[idx_bnd, 1],
        z_bnd[idx_bnd],
        c="red",
        s=14,
        alpha=0.9,
        label=rf"boundary points $\partial \Omega$",
    )

    ax.set_title("Sampling points")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$\psi$")
    ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_dir / "sampling_points_3d.png", dpi=300)
    plt.close()



    # ------------------------------------------------------------
    # Save 2D sampling points
    # green = interior points, red = boundary points
    # ------------------------------------------------------------
    x_int = torch.rand(cfg.n_interior, 2, device=cfg.device, dtype=cfg.dtype)
    x_bnd = make_boundary_points(cfg)

    x_int_np = x_int.detach().cpu().numpy()
    x_bnd_np = x_bnd.detach().cpu().numpy()

    # Optional: reduce displayed points for clarity
    n_show_int = min(2000, cfg.n_interior)
    n_show_bnd = min(1000, cfg.n_boundary)

    idx_int = np.random.choice(cfg.n_interior, n_show_int, replace=False)
    idx_bnd = np.random.choice(cfg.n_boundary, n_show_bnd, replace=False)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        x_int_np[idx_int, 0],
        x_int_np[idx_int, 1],
        c="green",
        s=9,
        alpha=0.7,
        label=rf"interior points $\Omega$",
        )

    plt.scatter(
        x_bnd_np[idx_bnd, 0],
        x_bnd_np[idx_bnd, 1],
        c="red",
        s=12,
        alpha=0.9,
        label=rf"boundary points $\partial \Omega$",
        )

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Sampling points")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize = 15)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "sampling_points_2d.png", dpi=300)
    plt.close()

    print(f"Relative L2 error  = {data['rel_l2']:.6e}")
    print(f"Relative Linf error = {data['rel_linf']:.6e}")
    print(f"Approximate free-boundary radius = {data['fb_radius']:.6f}")


def load_model_from_checkpoint(cfg: Config, checkpoint_path: Path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg.device,
    )

    if "config" in checkpoint:
        saved_cfg = checkpoint["config"]

        if "widths" in saved_cfg:
            cfg.widths = tuple(saved_cfg["widths"])

        if "grid_size" in saved_cfg:
            cfg.grid_size = int(saved_cfg["grid_size"])

    model = SquareObstacleKANPINN(
        widths=cfg.widths,
        grid_size=cfg.grid_size,
    ).to(cfg.device, cfg.dtype)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main():
    cfg = Config()
    out_dir = Path(cfg.out_dir)

    history_path = out_dir / "history.npz"
    checkpoint_path = out_dir / "kan_pinn_obstacle_dome_square.pt"

    if history_path.exists():
        save_training_curves(
            history_path=history_path,
            out_dir=out_dir,
            print_every=cfg.print_every,
        )
        print(f"Saved training curves to {out_dir.resolve()}")
    else:
        print(f"Could not find history file: {history_path}")

    if checkpoint_path.exists():
        model = load_model_from_checkpoint(cfg, checkpoint_path)
        save_solution_plots(model, cfg, out_dir)
        print(f"Saved solution plots to {out_dir.resolve()}")
    else:
        print(f"Could not find checkpoint file: {checkpoint_path}")


if __name__ == "__main__":
    main()