# plot_results_compare.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 20,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "legend.fontsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "lines.linewidth": 5.0,
    "grid.alpha": 0.7,
})


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path("outputs_dome_method_width_comparison")

RESULTS_NPZ = ROOT / "dome_method_width_comparison_results.npz"

METHOD_ORDER = ["PINN", "ResNet", "KAN"]

METHOD_STYLE = {
    "PINN": {
        "color": "tab:blue",
        "alpha": 0.45,
    },
    "ResNet": {
        "color": "tab:orange",
        "alpha": 0.45,
    },
    "KAN": {
        "color": "tab:green",
        "alpha": 0.45,
    },
}


# ------------------------------------------------------------
# Load global benchmark summary
# ------------------------------------------------------------

def load_summary_results(results_npz: Path):
    if not results_npz.exists():
        raise FileNotFoundError(f"Could not find summary NPZ file: {results_npz}")

    data = np.load(results_npz, allow_pickle=True)

    results = []

    for method, width, params, final_rel_l2, final_rel_linf in zip(
        data["method"],
        data["width"],
        data["params"],
        data["final_rel_l2"],
        data["final_rel_linf"],
    ):
        results.append({
            "method": str(method),
            "width": int(width),
            "params": int(params),
            "final_rel_l2": float(final_rel_l2),
            "final_rel_linf": float(final_rel_linf),
        })

    return results


# ------------------------------------------------------------
# Load each history file
# ------------------------------------------------------------

def history_path_for(root: Path, method: str, width: int):
    return root / method / f"width_{width}" / f"history_{method.lower()}_width_{width}.npz"


def load_rel_l2_history(root: Path, method: str, width: int):
    history_path = history_path_for(root, method, width)

    if not history_path.exists():
        raise FileNotFoundError(f"Could not find history file: {history_path}")

    hist = np.load(history_path, allow_pickle=True)

    if "rel_l2" not in hist.files:
        raise KeyError(f"'rel_l2' not found in {history_path}")

    rel_l2 = np.asarray(hist["rel_l2"], dtype=float)

    # Remove invalid entries if any.
    rel_l2 = rel_l2[np.isfinite(rel_l2)]
    rel_l2 = rel_l2[rel_l2 > 0]

    if len(rel_l2) == 0:
        raise ValueError(f"No valid positive rel_l2 values in {history_path}")

    return rel_l2


# ------------------------------------------------------------
# Optional bootstrap statistics
# ------------------------------------------------------------

def bootstrap_mean_std(values, n_boot=2000, seed=12345):
    """
    Bootstrap the mean of the logged relative L2 errors.

    Returns:
        sample_mean, sample_std, bootstrap_mean_std
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)

    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.asarray(boot_means)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "boot_mean_std": float(np.std(boot_means, ddof=1)) if len(boot_means) > 1 else 0.0,
    }


# ------------------------------------------------------------
# Build plotting records
# ------------------------------------------------------------

def build_plot_records(root: Path, summary_results):
    records = []

    for r in summary_results:
        method = r["method"]
        width = r["width"]
        params = r["params"]

        rel_l2_hist = load_rel_l2_history(root, method, width)
        stats = bootstrap_mean_std(rel_l2_hist)

        records.append({
            "method": method,
            "width": width,
            "params": params,
            "final_rel_l2": r["final_rel_l2"],
            "final_rel_linf": r["final_rel_linf"],
            "rel_l2_samples": rel_l2_hist,
            "mean_rel_l2": stats["mean"],
            "std_rel_l2": stats["std"],
            "boot_mean_std": stats["boot_mean_std"],
        })

    return records


# ------------------------------------------------------------
# Box-whisker plot
# ------------------------------------------------------------

def plot_box_whisker_by_params(records, out_dir: Path):
    fig, ax = plt.subplots(figsize=(15, 10))

    all_params = np.array([r["params"] for r in records], dtype=float)
    min_params = float(np.min(all_params))
    max_params = float(np.max(all_params))

    legend_handles = []

    for method in METHOD_ORDER:
        subset = [r for r in records if r["method"] == method]
        subset = sorted(subset, key=lambda z: z["params"])

        if len(subset) == 0:
            continue

        color = METHOD_STYLE[method]["color"]
        alpha = METHOD_STYLE[method]["alpha"]

        data = [r["rel_l2_samples"] for r in subset]
        positions = [r["params"] for r in subset]

        # Box width proportional to total trainable parameters.
        # On a log x-axis, using a fraction of the x-position keeps the visual
        # box size naturally scaled.
        box_widths = [0.12 * p for p in positions]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_widths,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            manage_ticks=True,
            whis=12.0,
        )

        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=alpha, edgecolor=color, linewidth=3)

        for median in bp["medians"]:
            median.set(color="black", linewidth=3)

        for mean in bp["means"]:
            mean.set(color=color, linewidth=3.5, linestyle="--")

        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=2.2)

        for cap in bp["caps"]:
            cap.set(color=color, linewidth=2.2)

        for flier in bp["fliers"]:
            flier.set(
                marker="o",
                markersize=4,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.35,
            )

        # Overlay final relative L2 error as diamond marker.
        ax.scatter(
            positions,
            [r["final_rel_l2"] for r in subset],
            marker="D",
            s=90,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )

        # Optional annotation: width above each box.
        for r in subset:
            ax.annotate(
                f"w={r['width']}",
                xy=(r["params"], r["final_rel_l2"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=12,
                color=color,
            )

        detail_lines = []
        for r in subset:
            detail_lines.append(
                f"#Width = {r['width']:d}, "
                f"Rel. err.={r['mean_rel_l2']:.1e} $\pm$ {r['std_rel_l2']:.1e}"
            )

        legend_label = method + "\n" + "\n".join(detail_lines)

        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                label=legend_label,
            )
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)

    # Use actual parameter values as ticks.
    #all_params_sorted = sorted(set(int(r["params"]) for r in records))
    #ax.set_xticks(all_params_sorted)
    #ax.set_xticklabels(
    #    [str(p) for p in all_params_sorted],
    #    rotation=35,
    #    ha="right",
    #)

    ax.set_xlabel("Total Trainable Parameters")
    ax.set_ylabel(r"Relative $L^2$ Error")
    ax.set_title("Obstacle Problem: Model Capacity")

    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    #ax.legend(
    #    handles=legend_handles,
    #    loc="upper center",
    #    bbox_to_anchor=(0.5, -0.25),
    #    ncol=1,
    #    frameon=True,
    #    fontsize=12,
    #)

    ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    frameon=True,
    fontsize=20,
    )

    fig.tight_layout()

    out_path = out_dir / "box_whisker_rel_l2_vs_params.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved box-whisker plot to: {out_path.resolve()}")


# ------------------------------------------------------------
# Mean +/- std plot
# ------------------------------------------------------------

def plot_mean_std_by_params(records, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in METHOD_ORDER:
        subset = [r for r in records if r["method"] == method]
        subset = sorted(subset, key=lambda z: z["params"])

        if len(subset) == 0:
            continue

        color = METHOD_STYLE[method]["color"]

        params = np.array([r["params"] for r in subset], dtype=float)
        mean = np.array([r["mean_rel_l2"] for r in subset], dtype=float)
        std = np.array([r["std_rel_l2"] for r in subset], dtype=float)

        lower = np.maximum(mean - std, 1e-16)
        upper = mean + std

        ax.plot(
            params,
            mean,
            marker="D",
            linestyle="-.",
            linewidth=3,
            markersize=8,
            color=color,
            label=method,
        )

        ax.fill_between(
            params,
            lower,
            upper,
            color=color,
            alpha=0.18,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)

    all_params_sorted = sorted(set(int(r["params"]) for r in records))
    ax.set_xticks(all_params_sorted)
    ax.set_xticklabels(
        [str(p) for p in all_params_sorted],
        rotation=35,
        ha="right",
    )

    ax.set_xlabel("Total Trainable Parameters")
    ax.set_ylabel(r"Relative $L^2$ Error")
    ax.set_title(r"Obstacle Problem: Mean $\pm$ Std of Relative $L^2$ Error")

    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=True,
        fontsize=14,
    )

    fig.tight_layout()

    out_path = out_dir / "mean_std_rel_l2_vs_params.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved mean/std plot to: {out_path.resolve()}")


# ------------------------------------------------------------
# Save summary table
# ------------------------------------------------------------

def save_summary_table(records, out_dir: Path):
    out_path = out_dir / "box_whisker_summary.csv"

    with open(out_path, "w") as f:
        f.write(
            "method,width,params,final_rel_l2,mean_rel_l2,std_rel_l2,"
            "bootstrap_std_of_mean,final_rel_linf\n"
        )

        for r in sorted(records, key=lambda z: (z["method"], z["params"])):
            f.write(
                f"{r['method']},"
                f"{r['width']},"
                f"{r['params']},"
                f"{r['final_rel_l2']:.16e},"
                f"{r['mean_rel_l2']:.16e},"
                f"{r['std_rel_l2']:.16e},"
                f"{r['boot_mean_std']:.16e},"
                f"{r['final_rel_linf']:.16e}\n"
            )

    print(f"Saved summary table to: {out_path.resolve()}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    out_dir = ROOT

    summary_results = load_summary_results(RESULTS_NPZ)
    records = build_plot_records(ROOT, summary_results)

    save_summary_table(records, out_dir)
    plot_box_whisker_by_params(records, out_dir)
    plot_mean_std_by_params(records, out_dir)


if __name__ == "__main__":
    main()