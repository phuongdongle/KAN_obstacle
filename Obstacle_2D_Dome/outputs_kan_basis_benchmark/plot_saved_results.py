from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def load_results_from_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    results = []

    for widths, grid_size, params, final_rel_l2, final_rel_linf in zip(
        data["widths"],
        data["grid_size"],
        data["params"],
        data["final_rel_l2"],
        data["final_rel_linf"],
    ):
        results.append(
            {
                "widths": str(widths),
                "grid_size": int(grid_size),
                "params": int(params),
                "final_rel_l2": float(final_rel_l2),
                "final_rel_linf": float(final_rel_linf),
            }
        )

    return results


def plot_benchmark(results, out_dir: Path):
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

        params_by_grid = [r["params"] for r in subset]
        params_text = "/".join(str(p) for p in params_by_grid)

        label = f"KAN {widths}"

        plt.plot(
            grid_sizes,
            rel_l2,
            "d-.",
            linewidth=3.5,
            markersize=12,
            label=label,
        )

    plt.yscale("log", base = 10)
    plt.xticks([5, 10, 15, 20])
    plt.xlabel("Grid size")
    plt.ylabel(r"Relative $L^2$ Error")
    plt.title("Obstacle Problem: Error vs. Grid-Size")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    #plt.legend(fontsize=15, ncol = 2, loc = "best")
    plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,          # or 4 if you want all entries in one row
    fontsize=14,
    frameon=True,
    )
    
    plt.tight_layout()

    out_path = out_dir / "kan_basis_benchmark_rel_l2_from_npz.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {out_path.resolve()}")


def main():
    #out_dir = Path("outputs_kan_basis_benchmark")
    out_dir = Path(".")
    npz_path = out_dir / "kan_basis_benchmark_results.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find NPZ file: {npz_path}")

    results = load_results_from_npz(npz_path)
    plot_benchmark(results, out_dir)


if __name__ == "__main__":
    main()