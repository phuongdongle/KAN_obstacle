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


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    xmin: float = 0.0
    xmax: float = 2.0
    ymin: float = 0.0
    ymax: float = 2.0

    n_eval_2d: int = 256

    # Defaults. These will be overwritten from checkpoint config if available.
    widths: tuple = (2, 64, 64, 64, 1)
    activation: str = "tanh"

    hidden_dim: int = 64
    n_blocks: int = 4


# ============================================================
# Problem definition
# ============================================================

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


# ============================================================
# Error diagnostics
# ============================================================

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


# ============================================================
# MLP-PINN model, soft boundary condition version
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


class MLPNet(nn.Module):
    def __init__(self, widths, activation: str = "tanh"):
        super().__init__()

        layers = []

        for din, dout in zip(widths[:-2], widths[1:-1]):
            layers.append(nn.Linear(din, dout))
            layers.append(get_activation(activation))

        layers.append(nn.Linear(widths[-2], widths[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquareObstacleMLPPINN(nn.Module):
    def __init__(self, widths, activation: str = "tanh"):
        super().__init__()
        self.mlp = MLPNet(widths=widths, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ============================================================
# ResNet-PINN model, soft boundary condition version
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.input_layer(x))

        for block in self.blocks:
            z = block(z)

        return self.output_layer(z)


class SquareObstacleResNetPINN(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        out_dim: int = 1,
        n_blocks: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()

        self.resnet = ResNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_blocks=n_blocks,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


# ============================================================
# Evaluation
# ============================================================

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
    max_abs_error = float(torch.max(ERR).detach().cpu())

    return {
        "X": X1.detach().cpu().numpy(),
        "Y": X2.detach().cpu().numpy(),
        "U_pred": U_pred.detach().cpu().numpy(),
        "U_true": U_true.detach().cpu().numpy(),
        "ERR": ERR.detach().cpu().numpy(),
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
        "max_abs_error": max_abs_error,
    }


# ============================================================
# Plotting
# ============================================================

def save_2d_field(
    X,
    Y,
    Z,
    title: str,
    cbar_label: str,
    out_path: Path,
    add_contours: bool = False,
    scientific_colorbar: bool = False,
):
    plt.figure(figsize=(6, 5))

    im = plt.pcolormesh(
        X,
        Y,
        Z.T,
        cmap="jet",
        shading="auto",
    )

    if add_contours:
        plt.contour(
            X,
            Y,
            Z.T,
            levels=15,
            colors="black",
            linewidths=0.5,
        )

    cbar = plt.colorbar(im, label=cbar_label)

    if scientific_colorbar:
        sf = mticker.ScalarFormatter(useMathText=True, useOffset=False)
        sf.set_powerlimits((0, 0))
        cbar.formatter = sf
        cbar.update_ticks()

    plt.gca().set_aspect("equal")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_solution_2d_plots(
    model: nn.Module,
    cfg: Config,
    save_dir: Path,
    model_label: str,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    data = evaluate_on_square(model, cfg)

    X = data["X"]
    Y = data["Y"]
    U_true = data["U_true"]
    U_pred = data["U_pred"]
    ERR = data["ERR"]

    save_2d_field(
        X,
        Y,
        U_true,
        title="Exact solution",
        cbar_label=r"$u_{\mathrm{exact}}$",
        out_path=save_dir / "u_exact_2d.png",
        add_contours=True,
    )

    save_2d_field(
        X,
        Y,
        U_pred,
        title=f"{model_label} approximation",
        cbar_label=rf"$u_{{\mathrm{{{model_label}}}}}$",
        out_path=save_dir / "u_pred_2d.png",
        add_contours=True,
    )

    save_2d_field(
        X,
        Y,
        ERR,
        title="Absolute error",
        cbar_label=rf"$|u_{{\mathrm{{{model_label}}}}}-u_{{\mathrm{{exact}}}}|$",
        out_path=save_dir / "absolute_error_2d.png",
        add_contours=False,
        scientific_colorbar=True,
    )

    print(f"\n{model_label} results")
    print(f"Saved plots to: {save_dir.resolve()}")
    print(f"Relative L2 error   = {data['rel_l2']:.6e}")
    print(f"Relative Linf error = {data['rel_linf']:.6e}")
    print(f"Max absolute error  = {data['max_abs_error']:.6e}")


# ============================================================
# Checkpoint loading
# ============================================================

def update_cfg_from_checkpoint(cfg: Config, checkpoint: dict):
    if "config" not in checkpoint:
        return cfg

    saved_cfg = checkpoint["config"]

    if "widths" in saved_cfg:
        cfg.widths = tuple(saved_cfg["widths"])

    if "activation" in saved_cfg:
        cfg.activation = saved_cfg["activation"]

    if "hidden_dim" in saved_cfg:
        cfg.hidden_dim = int(saved_cfg["hidden_dim"])

    if "n_blocks" in saved_cfg:
        cfg.n_blocks = int(saved_cfg["n_blocks"])

    if "n_eval_2d" in saved_cfg:
        cfg.n_eval_2d = int(saved_cfg["n_eval_2d"])

    if "xmin" in saved_cfg:
        cfg.xmin = float(saved_cfg["xmin"])

    if "xmax" in saved_cfg:
        cfg.xmax = float(saved_cfg["xmax"])

    if "ymin" in saved_cfg:
        cfg.ymin = float(saved_cfg["ymin"])

    if "ymax" in saved_cfg:
        cfg.ymax = float(saved_cfg["ymax"])

    return cfg


def load_mlp_model(checkpoint_path: Path, cfg: Config):
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    cfg = update_cfg_from_checkpoint(cfg, checkpoint)

    model = SquareObstacleMLPPINN(
        widths=cfg.widths,
        activation=cfg.activation,
    ).to(cfg.device, cfg.dtype)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg


def load_resnet_model(checkpoint_path: Path, cfg: Config):
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    cfg = update_cfg_from_checkpoint(cfg, checkpoint)

    model = SquareObstacleResNetPINN(
        in_dim=2,
        hidden_dim=cfg.hidden_dim,
        out_dim=1,
        n_blocks=cfg.n_blocks,
        activation=cfg.activation,
    ).to(cfg.device, cfg.dtype)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg


# ============================================================
# Main
# ============================================================

def main():
    # ------------------------------------------------------------
    # Your saved MLP-PINN paths
    # ------------------------------------------------------------
    mlp_root = Path("outputs_mlp_plap_obstacle_exact")
    mlp_history_path = mlp_root / "history_mlp_pinn_plap_obstacle.npz"
    mlp_checkpoint_path = mlp_root / "mlp_pinn_plap_obstacle.pt"
    mlp_save_dir = mlp_root / "PINN"

    # ------------------------------------------------------------
    # Your saved ResNet-PINN paths
    # ------------------------------------------------------------
    resnet_root = Path("outputs_resnet_plap_obstacle_exact")
    resnet_history_path = resnet_root / "history_resnet_pinn_plap_obstacle.npz"
    resnet_checkpoint_path = resnet_root / "resnet_pinn_plap_obstacle.pt"
    resnet_save_dir = resnet_root / "ResNet"

    # ------------------------------------------------------------
    # MLP-PINN plots
    # ------------------------------------------------------------
    if not mlp_history_path.exists():
        print(f"Warning: could not find MLP history file: {mlp_history_path}")

    if mlp_checkpoint_path.exists():
        cfg_mlp = Config()
        mlp_model, cfg_mlp = load_mlp_model(mlp_checkpoint_path, cfg_mlp)

        save_solution_2d_plots(
            model=mlp_model,
            cfg=cfg_mlp,
            save_dir=mlp_save_dir,
            model_label="PINN",
        )
    else:
        print(f"Could not find MLP checkpoint file: {mlp_checkpoint_path}")

    # ------------------------------------------------------------
    # ResNet-PINN plots
    # ------------------------------------------------------------
    if not resnet_history_path.exists():
        print(f"Warning: could not find ResNet history file: {resnet_history_path}")

    if resnet_checkpoint_path.exists():
        cfg_resnet = Config()
        resnet_model, cfg_resnet = load_resnet_model(
            resnet_checkpoint_path,
            cfg_resnet,
        )

        save_solution_2d_plots(
            model=resnet_model,
            cfg=cfg_resnet,
            save_dir=resnet_save_dir,
            model_label="ResNet",
        )
    else:
        print(f"Could not find ResNet checkpoint file: {resnet_checkpoint_path}")


if __name__ == "__main__":
    main()