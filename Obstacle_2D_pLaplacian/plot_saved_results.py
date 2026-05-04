from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "lines.linewidth": 3.0,
    "grid.alpha": 0.7,
})


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    xmin: float = 0.0
    xmax: float = 2.0
    ymin: float = 0.0
    ymax: float = 2.0

    widths: tuple = (2, 64, 64, 64, 1)
    grid_size: int = 20
    use_rbf_kan: bool = False

    print_every: int = 250
    n_eval_2d: int = 256

    out_dir: str = "outputs_kan_plap_obstacle_exact"


def psi_fn(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0]
    psi = torch.zeros_like(x1)
    inside = (x1 >= 0.5) & (x1 <= 1.5)
    return torch.where(inside, torch.ones_like(x1), psi)


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


class PiecewiseLinearBasis(nn.Module):
    def __init__(self, grid_size: int, x_min: float = 0.0, x_max: float = 2.0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        grid = torch.linspace(x_min, x_max, grid_size)
        self.register_buffer("grid", grid)
        self.h = (x_max - x_min) / (grid_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.x_min, self.x_max)
        return torch.relu(1.0 - torch.abs(x.unsqueeze(-1) - self.grid) / self.h)


class RBFBasis(nn.Module):
    def __init__(self, grid_size: int, x_min: float = 0.0, x_max: float = 2.0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        centers = torch.linspace(x_min, x_max, grid_size)
        self.register_buffer("centers", centers)
        h = (x_max - x_min) / (grid_size - 1)
        self.gamma = 1.0 / (h**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.x_min, self.x_max)
        diff = x.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff**2)


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int, use_rbf: bool = False):
        super().__init__()
        self.basis = RBFBasis(grid_size) if use_rbf else PiecewiseLinearBasis(grid_size)

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
            KANLayer(din, dout, grid_size, use_rbf=use_rbf)
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
    def __init__(self, widths, grid_size: int, use_rbf: bool = False):
        super().__init__()
        self.kan = KANNet(widths, grid_size, use_rbf=use_rbf)

    def boundary_factor(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        return x1 * (2.0 - x1) * x2 * (2.0 - x2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = g_fn(x).view(-1, 1)
        raw = self.kan(x)
        return g + self.boundary_factor(x) * raw


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

    x1 = torch.linspace(cfg.xmin, cfg.xmax, n, device=cfg.device, dtype=cfg.dtype)
    x2 = torch.linspace(cfg.ymin, cfg.ymax, n, device=cfg.device, dtype=cfg.dtype)

    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    pts = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

    U_pred = model(pts).reshape(n, n)
    U_true = u_exact_fn(pts).reshape(n, n)
    PSI = psi_fn(pts).reshape(n, n)
    ERR = torch.abs(U_pred - U_true)
    GAP = U_pred - PSI

    return {
        "X": X1.detach().cpu().numpy(),
        "Y": X2.detach().cpu().numpy(),
        "U_pred": U_pred.detach().cpu().numpy(),
        "U_true": U_true.detach().cpu().numpy(),
        "PSI": PSI.detach().cpu().numpy(),
        "ERR": ERR.detach().cpu().numpy(),
        "GAP": GAP.detach().cpu().numpy(),
        "rel_l2": relative_l2_error(U_pred.reshape(-1, 1), U_true.reshape(-1, 1)),
        "rel_linf": relative_linf_error(U_pred.reshape(-1, 1), U_true.reshape(-1, 1)),
        "max_abs_error": float(torch.max(ERR).detach().cpu()),
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
        shade=True,
        rstride=2,
        cstride=2,
    )

    # View shot
    ax.view_init(elev=25, azim=-130)
    
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    #ax.set_zlabel(zlabel)
    
    ax.set_title(title)

    cbar=fig.colorbar(surf, shrink=0.7, aspect=16)
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
    plt.plot(epochs_logged, history["total"], "k-", label="total loss")
    plt.plot(epochs_logged, history["obs"], "b--", label="obstacle loss")
    plt.plot(epochs_logged, history["pde"], "r--", label="PDE loss")
    plt.plot(epochs_logged, history["comp"], "g--", label="complementarity loss")

    #if "bc" in history.files:
    #    plt.plot(epochs_logged, history["bc"], "m-.", label="boundary loss")

    plt.yscale("log", base = 10)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")
    plt.ylim(1e-12, 1e3)
    #plt.title("Training loss components")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_components_logscale.png", dpi=300)
    plt.close()

    if "rel_l2" in history.files and "rel_linf" in history.files:
        plt.figure(figsize=(8, 5))
        plt.plot(
            epochs_logged,
            history["rel_l2"],
            "b--",
            label=r"$\|u_{\mathrm{exact}}-u_{\mathrm{KAN}}\|_{L^2}/\|u_{\mathrm{exact}}\|_{L^2}$",
        )
        plt.plot(
            epochs_logged,
            history["rel_linf"],
            "r-.",
            label=r"$\|u_{\mathrm{exact}}-u_{\mathrm{KAN}}\|_{L^\infty}/\|u_{\mathrm{exact}}\|_{L^\infty}$",
        )

        plt.yscale("log", base = 10)
        plt.xlabel("Number of epochs")
        plt.ylabel("Relative error")
        plt.ylim(1e-4, 1e0)
        plt.legend()
        
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
    GAP = data["GAP"]

    save_surface(X, Y, U_true, "Exact solution", r"$u_{\mathrm{exact}}$", out_dir / "u_exact_3d.png")
    save_surface(X, Y, U_pred, "KAN approximation", r"$u_{\mathrm{KAN}}$", out_dir / "u_pred_3d.png")
    save_surface(X, Y, PSI, "Obstacle", r"$\psi$", out_dir / "obstacle_3d.png")

    for arr, name, title, cbar_label in [
        (U_true, "u_exact_2d.png", "Exact solution", r"$u_{\mathrm{exact}}$"),
        (U_pred, "u_pred_2d.png", "KAN approximation", r"$u_{\mathrm{KAN}}$"),
        (PSI, "obstacle_2d.png", "Obstacle", r"$\psi$"),
        (ERR, "absolute_error_2d.png", "Absolute error", r"$|u_{\mathrm{KAN}}-u_{\mathrm{exact}}|$"),
        (GAP, "gap_function_2d.png", r"Gap function $u_{\mathrm{KAN}}-\psi$", r"$u_{\mathrm{KAN}}-\psi$"),
    ]:
        plt.figure(figsize=(6, 5))
        im = plt.pcolormesh(X, Y, arr.T, cmap="jet", shading="auto")

        if name in ["u_exact_2d.png", "u_pred_2d.png"]:
            plt.contour(X, Y, arr.T, levels=15, colors="black", linewidths=0.5)

        if name == "absolute_error_2d.png":
            cbar = plt.colorbar(im, label=cbar_label)
            sf = mticker.ScalarFormatter(useMathText=True, useOffset=False)
            sf.set_powerlimits((0, 0))
            cbar.formatter = sf
            cbar.update_ticks()
        else:
            plt.colorbar(im, label=cbar_label)

        plt.gca().set_aspect("equal")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=300)
        plt.close()

    print(f"Relative L2 error   = {data['rel_l2']:.6e}")
    print(f"Relative Linf error = {data['rel_linf']:.6e}")
    print(f"Max absolute error  = {data['max_abs_error']:.6e}")


def load_model_from_checkpoint(cfg: Config, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

    if "config" in checkpoint:
        saved_cfg = checkpoint["config"]

        if "widths" in saved_cfg:
            cfg.widths = tuple(saved_cfg["widths"])

        if "grid_size" in saved_cfg:
            cfg.grid_size = int(saved_cfg["grid_size"])

        if "use_rbf_kan" in saved_cfg:
            cfg.use_rbf_kan = bool(saved_cfg["use_rbf_kan"])

    model = SquareObstacleKANPINN(
        widths=cfg.widths,
        grid_size=cfg.grid_size,
        use_rbf=cfg.use_rbf_kan,
    ).to(cfg.device, cfg.dtype)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main():
    cfg = Config()
    out_dir = Path(cfg.out_dir)

    history_path = out_dir / "history.npz"
    checkpoint_path = out_dir / "kan_pinn_plap_obstacle.pt"

    if history_path.exists():
        save_training_curves(history_path, out_dir, cfg.print_every)
        print(f"Saved training plots to {out_dir.resolve()}")
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



