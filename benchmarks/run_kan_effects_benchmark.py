#!/usr/bin/env python3
"""
Unified GPU-ready KAN grid-size / model-width benchmark for the three
free-boundary PDE examples in this repository:

  1. Elastic obstacle problem
  2. p-Laplacian obstacle problem
  3. 2-D one-phase Stefan problem

The script reuses the model/training code already present in each problem
folder and produces the Appendix-C layout requested for the paper:

  * one 1 x 2 figure: elastic obstacle + p-Laplacian obstacle
  * one 1 x 1 figure: Stefan problem

For Stefan, the main appendix panel reports the relative L2 error of u.
The relative L2 error of the learned free boundary s(y,t) is also saved as
an additional diagnostic figure and in the CSV/NPZ summaries.

Output structure
----------------
Appendix_C_KAN_Effects/
    png/
        appendix_C_obstacle_1x2.png
        appendix_C_stefan_1x1.png
        stefan_free_boundary_grid_effect.png
    pdf/
        appendix_C_obstacle_1x2.pdf
        appendix_C_stefan_1x1.pdf
        stefan_free_boundary_grid_effect.pdf
    csv/
        kan_effects_results.csv
    npz/
        kan_effects_results.npz
    runs/
        dome/...
        plap/...
        stefan/...

GPU behavior
------------
By default --device auto is used. CUDA is selected automatically whenever
PyTorch sees an allocated NVIDIA GPU. On a Slurm GPU node this normally means
CUDA_VISIBLE_DEVICES is already set, so --gpu 0 selects the first allocated GPU.

Examples
--------
Run all full benchmarks on GPU when available:
    python benchmarks/run_kan_effects_benchmark.py --problems all

Require CUDA explicitly:
    python benchmarks/run_kan_effects_benchmark.py --problems all --device cuda

Run only Stefan:
    python benchmarks/run_kan_effects_benchmark.py --problems stefan --device cuda

Smoke test:
    python benchmarks/run_kan_effects_benchmark.py --problems all --quick

Regenerate figures only:
    python benchmarks/run_kan_effects_benchmark.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make progress messages appear promptly in Slurm logs even when Python is not
# launched with -u. This has no effect on numerical results.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass


# ============================================================
# Project / output layout
# ============================================================


def locate_project_root(start: Path) -> Path:
    """
    Find the repository root containing the three problem directories.

    This works whether the benchmark script is placed:
      * in the repository root,
      * in ROOT/benchmarks/,
      * deeper inside the repository, or
      * next to / above a KAN-Free-Boundary-PDE-main directory.
    """
    required = (
        "Obstacle_2D_Dome",
        "Obstacle_2D_pLaplacian",
        "Stefan_Problem",
    )

    # Search the script directory and every parent directory.
    for candidate in (start, *start.parents):
        if all((candidate / name).is_dir() for name in required):
            return candidate.resolve()

        # Backward compatibility with the earlier archive/folder name.
        nested = candidate / "KAN-Free-Boundary-PDE-main"
        if nested.is_dir() and all((nested / name).is_dir() for name in required):
            return nested.resolve()

    raise FileNotFoundError(
        "Could not locate the project root containing "
        "Obstacle_2D_Dome, Obstacle_2D_pLaplacian, and Stefan_Problem. "
        f"Benchmark script location: {start}"
    )


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = locate_project_root(SCRIPT_DIR)
DEFAULT_OUT = ROOT / "Appendix_C_KAN_Effects"

GRID_SIZES_DEFAULT = (5, 10, 15, 20)
WIDTHS_DEFAULT = (8, 16, 32, 48)

PROBLEM_LABELS = {
    "dome": "Elastic Obstacle Problem",
    "plap": r"$p$-Laplacian Obstacle Problem",
    "stefan": "Stefan Problem",
}

COLORS = {
    8: "tab:blue",
    16: "tab:orange",
    32: "tab:green",
    48: "tab:red",
}

MARKERS = {
    8: "d",
    16: "d",
    32: "d",
    48: "d",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "lines.linewidth": 3.0,
    "grid.alpha": 0.7,
})


# ============================================================
# Utility helpers
# ============================================================


def make_output_dirs(out_dir: Path) -> dict:
    """Create separate PNG, PDF, CSV, NPZ, and run directories."""
    dirs = {
        "root": out_dir,
        "png": out_dir / "png",
        "pdf": out_dir / "pdf",
        "csv": out_dir / "csv",
        "npz": out_dir / "npz",
        "runs": out_dir / "runs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find module: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def count_params(*models: torch.nn.Module) -> int:
    return sum(
        p.numel()
        for model in models
        for p in model.parameters()
        if p.requires_grad
    )


def parse_int_list(text: str) -> Tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def result_key(problem: str, width: int, grid_size: int) -> Tuple[str, int, int]:
    return problem, int(width), int(grid_size)


def resolve_device(requested: str, gpu_index: int) -> str:
    """Resolve auto/cuda/cpu to a concrete PyTorch device string."""
    requested = requested.lower()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False. "
                "Check that this job is running on a GPU node and that a CUDA-enabled "
                "PyTorch build is installed."
            )
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"GPU index {gpu_index} is invalid. Visible CUDA device count: "
                f"{torch.cuda.device_count()}"
            )
        return f"cuda:{gpu_index}"

    if requested == "auto":
        if torch.cuda.is_available():
            if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU index {gpu_index} is invalid. Visible CUDA device count: "
                    f"{torch.cuda.device_count()}"
                )
            return f"cuda:{gpu_index}"
        return "cpu"

    raise ValueError(f"Unknown device mode: {requested}")


def configure_torch(device: str, tf32: bool = False) -> None:
    """Configure CUDA execution without using mixed precision."""
    if not device.startswith("cuda"):
        return

    index = int(device.split(":")[1]) if ":" in device else 0
    torch.cuda.set_device(index)

    # Keep the benchmark in float32. TF32 is optional because it may slightly
    # change numerical results, especially for derivative-based PDE losses.
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(tf32)


def print_device_info(device: str, tf32: bool) -> None:
    print("Device:      ", device)

    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        print("GPU:         ", torch.cuda.get_device_name(idx))
        print("GPU memory:  ", f"{total_gb:.2f} GB")
        print("CUDA runtime:", torch.version.cuda)
        print("TF32:        ", tf32)
    else:
        print("GPU:          not used")


def clear_device_cache(device: str) -> None:
    """Release Python references/cached CUDA blocks between benchmark runs."""
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# ============================================================
# Results I/O
# ============================================================


def read_existing_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []

    rows: List[dict] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "problem": r["problem"],
                "width": int(r["width"]),
                "grid_size": int(r["grid_size"]),
                "params": int(r["params"]),
                "rel_l2": float(r["rel_l2"]),
                "rel_linf": float(r["rel_linf"]),
                "rel_s_l2": float(r.get("rel_s_l2", "nan")),
                "rel_s_linf": float(r.get("rel_s_linf", "nan")),
                "u_arch": r.get("u_arch", ""),
                "s_arch": r.get("s_arch", ""),
                "device": r.get("device", ""),
            })
    return rows


def save_results(rows: Sequence[dict], dirs: dict) -> None:
    rows = sorted(rows, key=lambda r: (r["problem"], r["width"], r["grid_size"]))

    csv_path = dirs["csv"] / "kan_effects_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem",
                "width",
                "grid_size",
                "params",
                "rel_l2",
                "rel_linf",
                "rel_s_l2",
                "rel_s_linf",
                "u_arch",
                "s_arch",
                "device",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    np.savez(
        dirs["npz"] / "kan_effects_results.npz",
        problem=np.array([r["problem"] for r in rows]),
        width=np.array([r["width"] for r in rows], dtype=int),
        grid_size=np.array([r["grid_size"] for r in rows], dtype=int),
        params=np.array([r["params"] for r in rows], dtype=int),
        rel_l2=np.array([r["rel_l2"] for r in rows], dtype=float),
        rel_linf=np.array([r["rel_linf"] for r in rows], dtype=float),
        rel_s_l2=np.array([r["rel_s_l2"] for r in rows], dtype=float),
        rel_s_linf=np.array([r["rel_s_linf"] for r in rows], dtype=float),
        u_arch=np.array([r["u_arch"] for r in rows]),
        s_arch=np.array([r["s_arch"] for r in rows]),
        device=np.array([r.get("device", "") for r in rows]),
    )


# ============================================================
# Quick-test configuration
# ============================================================


def maybe_quick_config(problem: str, cfg, quick: bool):
    if not quick:
        return cfg

    common = {
        "epochs": 2,
        "print_every": 1,
    }

    if problem == "dome":
        return replace(
            cfg,
            **common,
            n_interior=128,
            n_boundary=64,
            n_eval_2d=32,
        )

    if problem == "plap":
        return replace(
            cfg,
            **common,
            n_interior=128,
            n_boundary=64,
            n_eval_2d=32,
        )

    if problem == "stefan":
        return replace(
            cfg,
            **common,
            n_interior=128,
            n_boundary=64,
            n_free_boundary=64,
            n_initial=64,
            n_eval=16,
        )

    raise ValueError(problem)


# ============================================================
# Problem benchmarks
# ============================================================


def benchmark_dome(
    widths: Sequence[int],
    grids: Sequence[int],
    dirs: dict,
    epochs: int | None,
    quick: bool,
    existing_keys: set,
    device: str,
) -> Iterable[dict]:
    mod = load_module(
        "kan_dome_basis",
        ROOT / "Obstacle_2D_Dome" / "kan_pinn_obstacle_radial_Bspline_test.py",
    )

    for width in widths:
        for grid_size in grids:
            key = result_key("dome", width, grid_size)
            if key in existing_keys:
                print(f"[skip] dome width={width}, grid={grid_size}: already in CSV")
                continue

            run_dir = dirs["runs"] / "dome" / f"width_{width}" / f"grid_{grid_size}"

            cfg = replace(
                mod.Config(),
                device=device,
                widths=(2, width, width, width, 1),
                grid_size=grid_size,
                out_dir=str(run_dir),
            )

            if epochs is not None:
                cfg = replace(cfg, epochs=epochs)
            cfg = maybe_quick_config("dome", cfg, quick)

            print(f"\n[DOME] width={width}, grid={grid_size}, device={cfg.device}")
            result = mod.train_one_configuration(cfg)

            row = {
                "problem": "dome",
                "width": width,
                "grid_size": grid_size,
                "params": int(result["params"]),
                "rel_l2": float(result["final_rel_l2"]),
                "rel_linf": float(result["final_rel_linf"]),
                "rel_s_l2": float("nan"),
                "rel_s_linf": float("nan"),
                "u_arch": str(cfg.widths),
                "s_arch": "",
                "device": device,
            }

            del result
            clear_device_cache(device)
            yield row


def benchmark_plap(
    widths: Sequence[int],
    grids: Sequence[int],
    dirs: dict,
    epochs: int | None,
    quick: bool,
    existing_keys: set,
    device: str,
) -> Iterable[dict]:
    mod = load_module(
        "kan_plap",
        ROOT / "Obstacle_2D_pLaplacian" / "kan_pinn_obstacle_p_laplacian.py",
    )

    for width in widths:
        for grid_size in grids:
            key = result_key("plap", width, grid_size)
            if key in existing_keys:
                print(f"[skip] plap width={width}, grid={grid_size}: already in CSV")
                continue

            run_dir = dirs["runs"] / "plap" / f"width_{width}" / f"grid_{grid_size}"

            cfg = replace(
                mod.Config(),
                device=device,
                widths=(2, width, width, width, 1),
                grid_size=grid_size,
                out_dir=str(run_dir),
            )

            if epochs is not None:
                cfg = replace(cfg, epochs=epochs)
            cfg = maybe_quick_config("plap", cfg, quick)

            print(f"\n[p-LAPLACIAN] width={width}, grid={grid_size}, device={cfg.device}")
            model, _history = mod.train(cfg)
            final = mod.evaluate_on_square(model, cfg)

            row = {
                "problem": "plap",
                "width": width,
                "grid_size": grid_size,
                "params": count_params(model),
                "rel_l2": float(final["rel_l2"]),
                "rel_linf": float(final["rel_linf"]),
                "rel_s_l2": float("nan"),
                "rel_s_linf": float("nan"),
                "u_arch": str(cfg.widths),
                "s_arch": "",
                "device": device,
            }

            del model, _history, final
            clear_device_cache(device)
            yield row


def benchmark_stefan(
    widths: Sequence[int],
    grids: Sequence[int],
    dirs: dict,
    epochs: int | None,
    quick: bool,
    existing_keys: set,
    device: str,
) -> Iterable[dict]:
    mod = load_module(
        "kan_stefan",
        ROOT / "Stefan_Problem" / "kan_pinn_stefan.py",
    )

    for width in widths:
        for grid_size in grids:
            key = result_key("stefan", width, grid_size)
            if key in existing_keys:
                print(f"[skip] stefan width={width}, grid={grid_size}: already in CSV")
                continue

            run_dir = dirs["runs"] / "stefan" / f"width_{width}" / f"grid_{grid_size}"

            # Keep a single hidden-width control for the capacity sweep.
            # u(x,y,t): 3 inputs; s(y,t): 2 inputs.
            u_widths = (3, width, width, width, 1)
            s_widths = (2, width, width, 1)

            cfg = replace(
                mod.Config(),
                device=device,
                u_widths=u_widths,
                s_widths=s_widths,
                grid_size=grid_size,
                out_dir=str(run_dir),
            )

            if epochs is not None:
                cfg = replace(cfg, epochs=epochs)
            cfg = maybe_quick_config("stefan", cfg, quick)

            print(f"\n[STEFAN] width={width}, grid={grid_size}, device={cfg.device}")
            u_model, s_model, _history = mod.train(cfg)
            rel_u_l2, rel_u_linf, rel_s_l2, rel_s_linf = mod.evaluate(
                cfg, u_model, s_model
            )

            row = {
                "problem": "stefan",
                "width": width,
                "grid_size": grid_size,
                "params": count_params(u_model, s_model),
                "rel_l2": float(rel_u_l2),
                "rel_linf": float(rel_u_linf),
                "rel_s_l2": float(rel_s_l2),
                "rel_s_linf": float(rel_s_linf),
                "u_arch": str(u_widths),
                "s_arch": str(s_widths),
                "device": device,
            }

            del u_model, s_model, _history
            clear_device_cache(device)
            yield row


# ============================================================
# Plotting
# ============================================================


def plot_problem(ax, rows: Sequence[dict], problem: str) -> None:
    problem_rows = [r for r in rows if r["problem"] == problem]

    if not problem_rows:
        ax.text(
            0.5,
            0.5,
            "No benchmark data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    for width in sorted({r["width"] for r in problem_rows}):
        subset = sorted(
            [r for r in problem_rows if r["width"] == width],
            key=lambda r: r["grid_size"],
        )

        x = [r["grid_size"] for r in subset]
        y = [r["rel_l2"] for r in subset]

        if problem in ("dome", "plap"):
            arch = f"(2, {width}, {width}, {width}, 1)"
            label = f"KAN {arch}"
        else:
            label = f"KAN width {width}"

        ax.plot(
            x,
            y,
            marker=MARKERS.get(width, "d"),
            linestyle="-.",
            linewidth=2.7,
            markersize=7,
            color=COLORS.get(width, None),
            label=label,
        )

    ax.set_yscale("log")
    ax.set_xticks(sorted({r["grid_size"] for r in problem_rows}))
    ax.set_xlabel("Grid size")
    ax.set_ylabel(r"Relative $L^2$ Error")
    ax.set_title(f"{PROBLEM_LABELS[problem]}: Error vs. Grid-Size")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=10, ncol=2)


def save_figure_both(fig, dirs: dict, stem: str, dpi: int = 300) -> None:
    fig.savefig(
        dirs["png"] / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        dirs["pdf"] / f"{stem}.pdf",
        bbox_inches="tight",
    )


def make_appendix_figures(rows: Sequence[dict], dirs: dict) -> None:
    # ------------------------------------------------------------
    # Appendix panel group 1: obstacle-type problems, 1 row x 2 cols
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

    plot_problem(axes[0], rows, "dome")
    plot_problem(axes[1], rows, "plap")

    axes[0].text(
        0.5,
        -0.30,
        "(a) Elastic Obstacle Problem",
        ha="center",
        va="top",
        transform=axes[0].transAxes,
        fontsize=16,
    )
    axes[1].text(
        0.5,
        -0.30,
        r"(b) $p$-Laplacian Obstacle Problem",
        ha="center",
        va="top",
        transform=axes[1].transAxes,
        fontsize=16,
    )

    fig.subplots_adjust(bottom=0.30, wspace=0.28)
    save_figure_both(fig, dirs, "appendix_C_obstacle_1x2")
    plt.close(fig)

    # ------------------------------------------------------------
    # Appendix panel group 2: Stefan problem, standalone 1 x 1
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    plot_problem(ax, rows, "stefan")

    ax.text(
        0.5,
        -0.27,
        "(c) Stefan Problem",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=16,
    )

    fig.subplots_adjust(bottom=0.25)
    save_figure_both(fig, dirs, "appendix_C_stefan_1x1")
    plt.close(fig)

    # ------------------------------------------------------------
    # Additional Stefan diagnostic: free-boundary s(y,t) error
    # ------------------------------------------------------------
    stefan_rows = [r for r in rows if r["problem"] == "stefan"]

    if stefan_rows and all(np.isfinite(r["rel_s_l2"]) for r in stefan_rows):
        fig, ax = plt.subplots(figsize=(8.2, 6.0))

        for width in sorted({r["width"] for r in stefan_rows}):
            subset = sorted(
                [r for r in stefan_rows if r["width"] == width],
                key=lambda r: r["grid_size"],
            )

            ax.plot(
                [r["grid_size"] for r in subset],
                [r["rel_s_l2"] for r in subset],
                marker="d",
                linestyle="-.",
                linewidth=2.7,
                markersize=7,
                color=COLORS.get(width, None),
                label=f"KAN width {width}",
            )

        ax.set_yscale("log")
        ax.set_xticks(sorted({r["grid_size"] for r in stefan_rows}))
        ax.set_xlabel("Grid size")
        ax.set_ylabel(r"Relative $L^2$ Error of $s$")
        ax.set_title("Stefan Problem: Free-Boundary Error vs. Grid-Size")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(fontsize=10, ncol=2)
        fig.tight_layout()

        save_figure_both(fig, dirs, "stefan_free_boundary_grid_effect")
        plt.close(fig)


def normalize_problem_list(values: Sequence[str]) -> List[str]:
    if "all" in values:
        return ["dome", "plap", "stefan"]

    out: List[str] = []
    for v in values:
        if v not in ("dome", "plap", "stefan"):
            raise ValueError(f"Unknown problem '{v}'")
        if v not in out:
            out.append(v)
    return out


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GPU-ready KAN grid-size/model-width study for the elastic obstacle, "
            "p-Laplacian obstacle, and Stefan free-boundary problems."
        )
    )

    parser.add_argument(
        "--problems",
        nargs="+",
        default=["all"],
        choices=["all", "dome", "plap", "stefan"],
        help="Problems to benchmark. Default: all.",
    )
    parser.add_argument(
        "--grids",
        type=parse_int_list,
        default=GRID_SIZES_DEFAULT,
        help="Comma-separated grid sizes. Default: 5,10,15,20",
    )
    parser.add_argument(
        "--widths",
        type=parse_int_list,
        default=WIDTHS_DEFAULT,
        help="Comma-separated hidden widths. Default: 8,16,32,48",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs defined in the original problem scripts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=(
            "Execution device. 'auto' uses CUDA when available, otherwise CPU. "
            "Use 'cuda' to require a GPU. Default: auto."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help=(
            "Visible CUDA device index. On Slurm, CUDA_VISIBLE_DEVICES usually "
            "maps the allocated GPU to index 0. Default: 0."
        ),
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help=(
            "Allow TF32 matrix math on supported NVIDIA GPUs. This can be faster "
            "but may slightly change numerical results. Disabled by default."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny 2-epoch smoke test with reduced collocation/evaluation grids.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Do not train; regenerate figures from csv/kan_effects_results.csv.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing CSV entries and rerun requested configurations.",
    )

    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    dirs = make_output_dirs(out_dir)

    csv_path = dirs["csv"] / "kan_effects_results.csv"
    rows = [] if args.restart else read_existing_csv(csv_path)

    # Plot-only mode does not need CUDA.
    if args.plot_only:
        if not rows:
            raise FileNotFoundError(
                f"No saved benchmark results found at {csv_path}. Run training first."
            )

        make_appendix_figures(rows, dirs)
        print(f"Figures regenerated in:\n  PNG: {dirs['png']}\n  PDF: {dirs['pdf']}")
        return

    device = resolve_device(args.device, args.gpu)
    configure_torch(device, tf32=args.tf32)

    problems = normalize_problem_list(args.problems)
    existing_keys = {
        result_key(r["problem"], r["width"], r["grid_size"])
        for r in rows
    }

    runners = {
        "dome": benchmark_dome,
        "plap": benchmark_plap,
        "stefan": benchmark_stefan,
    }

    print("=" * 88)
    print("KAN EFFECTS BENCHMARK")
    print("Project root:", ROOT)
    print("Output root: ", out_dir)
    print("PNG folder:  ", dirs["png"])
    print("PDF folder:  ", dirs["pdf"])
    print("CSV folder:  ", dirs["csv"])
    print("NPZ folder:  ", dirs["npz"])
    print_device_info(device, args.tf32)
    print("Problems:    ", ", ".join(problems))
    print("Widths:      ", tuple(args.widths))
    print("Grid sizes:  ", tuple(args.grids))
    print("Quick mode:  ", args.quick)
    print("=" * 88)

    for problem in problems:
        for row in runners[problem](
            widths=args.widths,
            grids=args.grids,
            dirs=dirs,
            epochs=args.epochs,
            quick=args.quick,
            existing_keys=existing_keys,
            device=device,
        ):
            key = result_key(row["problem"], row["width"], row["grid_size"])

            # Replace duplicates safely.
            rows = [
                r
                for r in rows
                if result_key(r["problem"], r["width"], r["grid_size"]) != key
            ]
            rows.append(row)
            existing_keys.add(key)

            # Save after every completed configuration so long GPU jobs resume safely.
            save_results(rows, dirs)
            print(f"[saved] {key} -> {csv_path}")

    save_results(rows, dirs)
    make_appendix_figures(rows, dirs)

    print("\nDone.")
    print(f"CSV:  {dirs['csv'] / 'kan_effects_results.csv'}")
    print(f"NPZ:  {dirs['npz'] / 'kan_effects_results.npz'}")
    print(f"PNG:  {dirs['png']}")
    print(f"PDF:  {dirs['pdf']}")
    print(f"Runs: {dirs['runs']}")


if __name__ == "__main__":
    main()
