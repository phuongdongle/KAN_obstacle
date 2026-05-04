from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm


dark_blues = LinearSegmentedColormap.from_list(
    "dark_blues",
    ["#dbeafe", "#3b82f6", "#1e3a8a", "#0f172a"]
)



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

    x_max: float = 2.25
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    u_widths: tuple = (3, 48, 48, 48, 1)
    s_widths: tuple = (2, 32, 32, 1)
    grid_size: int = 15

    n_plot_x: int = 300
    n_plot_y: int = 160
    print_every: int = 250

    out_dir: str = "outputs_kan_stefan_2d"


def u_exact_fn(xyt: torch.Tensor) -> torch.Tensor:
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    return torch.exp(1.25 * t - x + 0.5 * y + 0.5) - 1.0


def s_exact_fn(yt: torch.Tensor) -> torch.Tensor:
    y = yt[:, 0]
    t = yt[:, 1]
    return 0.5 * y + 1.25 * t + 0.5


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

        g_left = torch.exp(1.25 * t + 0.5 * y + 0.5) - 1.0
        g_bottom = torch.exp(1.25 * t - x + 0.5) - 1.0
        g_top = torch.exp(1.25 * t - x + 1.0) - 1.0

        c_bottom = torch.exp(1.25 * t + 0.5) - 1.0
        c_top = torch.exp(1.25 * t + 1.0) - 1.0

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

        return s0 + t * self.raw_net(yt)

@torch.no_grad()
def evaluate_snapshot(cfg, u_model, s_model, t_value):
    x = torch.linspace(0.0, cfg.x_max, cfg.n_plot_x, device=cfg.device, dtype=cfg.dtype)
    y = torch.linspace(0.0, 1.0, cfg.n_plot_y, device=cfg.device, dtype=cfg.dtype)

    X, Y = torch.meshgrid(x, y, indexing="xy")
    T = torch.full_like(X, float(t_value))

    pts = torch.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], dim=1)
    yt = torch.stack([Y.reshape(-1), T.reshape(-1)], dim=1)

    U_exact = u_exact_fn(pts).reshape(cfg.n_plot_y, cfg.n_plot_x)
    U_pred = u_model(pts).reshape(cfg.n_plot_y, cfg.n_plot_x)

    S_exact = s_exact_fn(yt).reshape(cfg.n_plot_y, cfg.n_plot_x)
    mask = X <= S_exact

    U_exact = U_exact.detach().cpu().numpy()
    U_pred = U_pred.detach().cpu().numpy()
    ERR = np.abs(U_pred - U_exact)

    mask_np = mask.detach().cpu().numpy()
    U_exact[~mask_np] = np.nan
    U_pred[~mask_np] = np.nan
    ERR[~mask_np] = np.nan

    return (
        X.detach().cpu().numpy(),
        Y.detach().cpu().numpy(),
        U_exact,
        U_pred,
        ERR,
    )


def plot_time_panel(cfg, u_model, s_model, out_dir):
    save_dir = out_dir / "TimeDependent"

    exact_dir = save_dir / "exact_snapshot"
    kan_dir = save_dir / "kan_snapshot"
    err_dir = save_dir / "err_snapshot"

    exact_dir.mkdir(parents=True, exist_ok=True)
    kan_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)

    times = [0.2, 0.4, 0.6, 0.8]

    for t_value in times:
        X, Y, U_exact, U_pred, ERR = evaluate_snapshot(
            cfg, u_model, s_model, t_value
        )

        snapshots = [
            (
                U_exact,
                exact_dir / f"exact_snapshot_{t_value:.1f}.png",
                rf"Exact $u(x_1,x_2,{t_value:.1f})$",
                r"$u_{\mathrm{exact}}$",
            ),
            (
                U_pred,
                kan_dir / f"kan_snapshot_{t_value:.1f}.png",
                rf"$u_{{\mathrm{{KAN}}}}(x_1,x_2,{t_value:.1f})$",
                r"$u_{\mathrm{KAN}}$",
            ),
            (
                ERR,
                err_dir / f"err_snapshot_{t_value:.1f}.png",
                "Absolute Error",
                r"$|u_{\mathrm{KAN}}-u_{\mathrm{exact}}|$",
            ),
        ]

        for arr, out_path, title, cbar_label in snapshots:
            fig, ax = plt.subplots(figsize=(6,5))

            im = ax.pcolormesh(
                X,
                Y,
                arr,
                cmap="jet",
                shading="auto",
            )

            ax.set_title(title)
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_xlim(0.0, cfg.x_max)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("auto")

            cbar = fig.colorbar(im, ax=ax)


            if "err_snapshot" in str(out_path):
                sf = mticker.ScalarFormatter(useMathText=True, useOffset=False)
                sf.set_powerlimits((0, 0))
                cbar.locator = mticker.LinearLocator(numticks=6)
                cbar.formatter = sf
                cbar.update_ticks()
            
            cbar.set_label(cbar_label, fontsize=15)
            cbar.ax.tick_params(labelsize=15)

            plt.tight_layout()
            fig.savefig(out_path, dpi=300)
            plt.close(fig)


def plot_free_surface_panel(cfg, u_model, s_model, out_dir):
    save_dir = out_dir / "TimeDependent" / "free_surface"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_y = 256
    n_t = 256

    y = torch.linspace(0.0, 1.0, n_y, device=cfg.device, dtype=cfg.dtype)
    t = torch.linspace(0.0, 1.0, n_t, device=cfg.device, dtype=cfg.dtype)

    Y, T = torch.meshgrid(y, t, indexing="xy")
    yt = torch.stack([Y.reshape(-1), T.reshape(-1)], dim=1)

    with torch.no_grad():
        S_pred = s_model(yt).reshape(n_t, n_y)
        S_exact = s_exact_fn(yt).reshape(n_t, n_y)
        S_err = torch.log10(torch.abs(S_exact - S_pred))

    Y_np = Y.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()
    S_pred_np = S_pred.detach().cpu().numpy()
    S_exact_np = S_exact.detach().cpu().numpy()
    S_err_np = S_err.detach().cpu().numpy()

    plots = [
        (S_exact_np, "Exact", "free_surface_exact.png", r"$s(x_2,t)$"),
        (S_pred_np, "Predicted", "free_surface_predicted.png", r"$\hat{s}(x_2,t)$"),
        (S_err_np, "Absolute Error", "free_surface_log_absolute_error.png",
         r"$\log_{10}(|s_{\mathrm{KAN}}-s_{\mathrm{exact}}|)$"),
        ]

    for Z, title, filename, zlabel in plots:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        surf=ax.plot_surface(
            Y_np,
            T_np,
            Z,
            cmap="inferno",
            linewidth=0,
            antialiased=True,
            shade=True,
            rstride=2,
            cstride=2,
        )

        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.65, aspect=14, pad=0.10)
        cbar.ax.set_title(zlabel, fontsize=18, pad=5)
        


        if "absolute_error" in filename:
            sf = mticker.ScalarFormatter(useMathText=True, useOffset=False)
            sf.set_powerlimits((0, 0))
            #cbar.locator = mticker.LinearLocator(numticks=6)
            cbar.locator = mticker.MaxNLocator(integer=True, nbins=6)
            cbar.formatter = sf
            cbar.update_ticks()

        ax.view_init(elev=28, azim=-45)
        


        
        # ---- FIX TICKS ----
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(4))

        ax.tick_params(axis='x', pad=1, labelsize=11)
        ax.tick_params(axis='y', pad=1, labelsize=11)
        ax.tick_params(axis='z', pad=1, labelsize=11)

        # Optional: cleaner formatting
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # ---- LABELS ----
        ax.set_title(title, fontsize=15, pad=2)
        ax.set_xlabel(r"$x_2$", fontsize=11, labelpad=1)
        ax.set_ylabel(r"$t$", fontsize=11, labelpad=1)
        #ax.set_zlabel(zlabel, fontsize=11, labelpad=6)

        
        fig.savefig(save_dir / filename, dpi=300)
        plt.close(fig)

    rel_s_l2 = np.linalg.norm(S_pred_np - S_exact_np) / np.linalg.norm(S_exact_np)
    print(f"Free-surface relative L2 error: {rel_s_l2:.6e}")
    print(f"Saved free-surface plots to: {save_dir}")




def plot_training_logs(cfg, out_dir):
    history_path = out_dir / "history.npz"
    if not history_path.exists():
        print(f"Could not find {history_path}")
        return

    save_dir = out_dir / "TimeDependent"
    save_dir.mkdir(parents=True, exist_ok=True)

    history = np.load(history_path)
    n_log = len(history["total"])
    epochs = np.arange(n_log) * cfg.print_every + 1
    if n_log > 1:
        epochs[1:] = np.arange(1, n_log) * cfg.print_every

    plt.figure(figsize=(8, 5))
    for key, label in [
        ("total", "total loss"),
        ("pde", "PDE loss"),
        ("ic", "initial loss"),
        ("fb_u", "free-boundary loss"),
        ("fb_stefan", "Stefan loss"),
    ]:
        if key in history.files:
            plt.plot(epochs, history[key], label=label, linewidth = 3)

    plt.yscale("log", base = 10)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")
    plt.legend(ncol = 2)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_dir / "training_losses.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for key, label, line_style in [
        ("rel_u_l2", r"$\|u_{\mathrm{exact}}-u_{\mathrm{KAN}}\|_{L^2}/\|u_{\mathrm{exact}}\|_{L^2}$", 'b--'),
        ("rel_u_linf", r"$\|u_{\mathrm{exact}}-u_{\mathrm{KAN}}\|_{L^\infty}/\|u_{\mathrm{exact}}\|_{L^\infty}$", 'r-.'),
    ]:
        if key in history.files:
            plt.plot(epochs, history[key], 
                     line_style,
                     label=label, 
                     linewidth = 3)

    plt.yscale("log", base = 10)
    plt.xlabel("Number of epochs")
    plt.ylabel("Relative error")
    plt.legend()
    plt.ylim(1e-3, 5e-2)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_dir / "relative_errors.png", dpi=300)
    plt.close()

def load_models(cfg, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg.device,
    )

    if "config" in checkpoint:
        saved_cfg = checkpoint["config"]
        cfg.u_widths = tuple(saved_cfg.get("u_widths", cfg.u_widths))
        cfg.s_widths = tuple(saved_cfg.get("s_widths", cfg.s_widths))
        cfg.grid_size = int(saved_cfg.get("grid_size", cfg.grid_size))
        cfg.x_max = float(saved_cfg.get("x_max", cfg.x_max))

    u_model = HardBCStefanUModel(
        widths=cfg.u_widths,
        grid_size=cfg.grid_size,
        x_max=cfg.x_max,
    ).to(cfg.device, cfg.dtype)

    s_model = HardICStefanSModel(
        widths=cfg.s_widths,
        grid_size=cfg.grid_size,
    ).to(cfg.device, cfg.dtype)

    u_model.load_state_dict(checkpoint["u_model_state_dict"])
    s_model.load_state_dict(checkpoint["s_model_state_dict"])

    u_model.eval()
    s_model.eval()

    return u_model, s_model

def compute_animation_ranges(cfg, u_model, s_model, n_sample_times=15):
    """
    Compute fixed color/z ranges for the animation over t in [0, 1].
    This avoids changing color scales from frame to frame.
    """
    times = np.linspace(cfg.t_min, cfg.t_max, n_sample_times)

    u_min, u_max = np.inf, -np.inf
    err_min, err_max = np.inf, -np.inf

    for t_value in times:
        _, _, _, U_pred, ERR = evaluate_snapshot(cfg, u_model, s_model, t_value)

        u_min = min(u_min, np.nanmin(U_pred))
        u_max = max(u_max, np.nanmax(U_pred))

        err_min = min(err_min, np.nanmin(ERR))
        err_max = max(err_max, np.nanmax(ERR))

    return u_min, u_max, err_min, err_max


def save_upred_error_panel_gif(cfg, u_model, s_model, out_dir):
    """
    Save a 1x2 panel GIF over t in [0, 1].

    Left panel:
        3D surface plot of u_KAN(x1, x2, t).

    Right panel:
        2D heatmap of absolute error |u_KAN - u_exact|.
    """

    save_dir = out_dir / "TimeDependent" / "gif"
    save_dir.mkdir(parents=True, exist_ok=True)

    gif_path = save_dir / "kan_stefan_upred_error_panel.gif"

    n_frames = 60
    fps = 10
    times = np.linspace(cfg.t_min, cfg.t_max, n_frames)

    u_min, u_max, err_min, err_max = compute_animation_ranges(
        cfg, u_model, s_model, n_sample_times=15
    )

    # First frame
    X, Y, _, U_pred, ERR = evaluate_snapshot(
        cfg, u_model, s_model, times[0]
    )

    fig = plt.figure(figsize=(13, 5.5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Dummy mappables for fixed colorbars
    u_norm = plt.Normalize(vmin=u_min, vmax=u_max)
    err_norm = plt.Normalize(vmin=err_min, vmax=err_max)

    u_mappable = cm.ScalarMappable(norm=u_norm, cmap="jet")
    u_mappable.set_array([])

    err_mappable = cm.ScalarMappable(norm=err_norm, cmap="jet")
    err_mappable.set_array([])

    cbar1 = fig.colorbar(
        u_mappable,
        ax=ax1,
        shrink=0.65,
        aspect=14,
        pad=0.08,
    )
    cbar1.ax.set_title(r"$u_{\mathrm{KAN}}$", fontsize=13, pad=5)
    cbar1.ax.tick_params(labelsize=12)
    

    cbar2 = fig.colorbar(
        err_mappable,
        ax=ax2,
        shrink=0.65,
        aspect=14,
        pad=0.10,
    )
    cbar2.set_label(
        r"$|u_{\mathrm{KAN}}-u_{\mathrm{exact}}|$",
        fontsize=14,
    )
    cbar2.ax.tick_params(labelsize=12)

    sf = mticker.ScalarFormatter(useMathText=True, useOffset=False)
    sf.set_powerlimits((0, 0))
    cbar2.formatter = sf
    cbar2.update_ticks()

    def update(frame):
        t_value = times[frame]

        X, Y, _, U_pred, ERR = evaluate_snapshot(
            cfg, u_model, s_model, t_value
        )

        ax1.clear()
        ax2.clear()

        # -------------------------
        # Left: 3D U_pred surface
        # -------------------------
        ax1.plot_surface(
            X,
            Y,
            U_pred,
            cmap="jet",
            linewidth=0,
            antialiased=True,
            shade=True,
            rstride=2,
            cstride=2,
            vmin=u_min,
            vmax=u_max,
        )

        ax1.set_title(
            rf"$u_{{\mathrm{{KAN}}}}(x_1,x_2,t)$, $t={t_value:.2f}$",
            fontsize=14,
            pad=8,
        )
        ax1.set_xlabel(r"$x_1$", fontsize=12, labelpad=5)
        ax1.set_ylabel(r"$x_2$", fontsize=12, labelpad=5)
        ax1.set_zlabel(r"$u_{\mathrm{KAN}}$", fontsize=12, labelpad=6)

        ax1.set_xlim(0.0, cfg.x_max)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_zlim(u_min, u_max)

        ax1.view_init(elev=28, azim=-45)

        ax1.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax1.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax1.zaxis.set_major_locator(mticker.MaxNLocator(4))

        ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax1.zaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        ax1.tick_params(axis="x", labelsize=10, pad=1)
        ax1.tick_params(axis="y", labelsize=10, pad=1)
        ax1.tick_params(axis="z", labelsize=10, pad=1)

        # -------------------------
        # Right: 2D absolute error
        # -------------------------
        im = ax2.pcolormesh(
            X,
            Y,
            ERR,
            cmap="jet",
            shading="auto",
            vmin=err_min,
            vmax=err_max,
        )

        ax2.set_title(
            rf"$|u_{{\mathrm{{KAN}}}}-u_{{\mathrm{{exact}}}}|$, $t={t_value:.2f}$",
            fontsize=14,
            pad=8,
        )
        ax2.set_xlabel(r"$x_1$", fontsize=12)
        ax2.set_ylabel(r"$x_2$", fontsize=12)

        ax2.set_xlim(0.0, cfg.x_max)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_aspect("auto")

        ax2.tick_params(axis="both", labelsize=11)

        fig.tight_layout()

        return im,

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=100,
        blit=False,
    )

    ani.save(
        gif_path,
        writer=PillowWriter(fps=fps),
        dpi=150,
    )

    plt.close(fig)

    print(f"Saved 1x2 panel GIF to: {gif_path}")



def main():
    cfg = Config()
    out_dir = Path(cfg.out_dir)
    checkpoint_path = out_dir / "kan_pinn_stefan_2d.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {checkpoint_path}")

    u_model, s_model = load_models(cfg, checkpoint_path)

    # Plot Time panel 
    plot_time_panel(cfg, u_model, s_model, out_dir) # as snapshots. 
    #save_upred_error_panel_gif(cfg, u_model, s_model, out_dir) # as videoframe GIF
    
    plot_free_surface_panel(cfg, u_model, s_model, out_dir)
    plot_training_logs(cfg, out_dir)

    print(f"Saved Stefan plots to: {out_dir / 'TimeDependent'}")


if __name__ == "__main__":
    main()