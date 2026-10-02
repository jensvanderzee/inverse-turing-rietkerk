"""Plot learned PDE parameters across all real-data training runs.

Mirrors `compare_invPDE_synthetic_params.py`, but for the real-data results
in `results/real_data_rietkerk/models/parameters`. Since there is no ground truth for the
real data, no reference lines are drawn and no relative-error metrics are
computed; the summary reports mean, std, and coefficient of variation.

Reads each model_XX_params.pkl (a list of snapshot dicts, each with an
'epoch' key and one entry per learned parameter) and:
  1. Prints per-run final learned values.
  2. Prints a summary table (mean, std, CV) across runs.
  3. Plots parameter trajectories vs epoch for every run.

Run from the repo root:
    python compare_invPDE_realdata_params.py
"""
#%%
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from rietkerk_model import (PARAM_NAMES, PRETTY_NAMES, REALDATA_REFERENCE,
                            to_physical_units, degenerate_parameters)

RESULTS_DIR = os.path.join("results", "real_data_rietkerk", "models", "parameters")
OUT_DIR = os.path.join("results", "real_data_rietkerk", "param_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# Tier 1 filter thresholds (applied to raw, un-scaled values).
LENGTH_FRAC = 0.9      # drop runs with fewer than 90% of the max snapshot count
# Runs with any final param NaN/inf or > 4 decades from the reference are dropped too
# (rietkerk_model.degenerate_parameters).

# Unit conversion for display and reporting: Rietkerk's physical units (m²/day
# for diffusion, g/m² for biomass, see rietkerk_model.to_physical_units) instead
# of pixel²/day and NDVI x 1500. The filter above still operates on raw values.
_PHYSICAL = to_physical_units(REALDATA_REFERENCE)
DISPLAY_SCALE = {name: _PHYSICAL[name] / REALDATA_REFERENCE[name] for name in PARAM_NAMES}


def scale(name, value):
    """Return value in display units (raw * DISPLAY_SCALE[name] if present)."""
    return value * DISPLAY_SCALE.get(name, 1.0)

#%%
def load_runs():
    paths = sorted(glob(os.path.join(RESULTS_DIR, "model_*_params.pkl")))
    if not paths:
        raise FileNotFoundError(f"No model_*_params.pkl files found in {RESULTS_DIR}")
    runs = []
    for p in paths:
        stem = os.path.basename(p).replace("model_", "").replace("_params.pkl", "")
        try:
            run_id = int(stem)
        except ValueError:
            continue
        with open(p, "rb") as f:
            history = pickle.load(f)
        if not history:
            continue
        runs.append({"run_id": run_id, "parameter_history": history})
    return runs


def final_params(run):
    last = run["parameter_history"][-1]
    return {k: last[k] for k in PARAM_NAMES if k in last}


def filter_tier1(runs):
    """Tier 1 filter: drop structural failures.

    Returns (kept, dropped) where dropped is a list of
    (run_id, tier, reason) tuples.
    """
    max_len = max(len(r["parameter_history"]) for r in runs)
    min_len = LENGTH_FRAC * max_len

    kept, dropped = [], []
    for r in runs:
        history = r["parameter_history"]
        reasons = []

        if len(history) < min_len:
            reasons.append(
                f"short history ({len(history)} < {min_len:.0f} snapshots)"
            )

        fp = final_params(r)
        bad = [f"{name}={fp.get(name, np.nan):.2e}"
               for name in degenerate_parameters(fp, REALDATA_REFERENCE)]
        if bad:
            reasons.append("degenerate final value(s): " + ", ".join(bad))

        if reasons:
            dropped.append((r["run_id"], 1, "; ".join(reasons)))
        else:
            kept.append(r)
    return kept, dropped


def print_dropped(dropped):
    if not dropped:
        print("Tier 1: no runs dropped.\n")
        return
    print("=" * 78)
    print(f"Tier 1: dropped {len(dropped)} runs")
    print(f"{'run':>4}  {'tier':>4}  reason")
    print("-" * 78)
    for run_id, tier, reason in dropped:
        print(f"{run_id:>4d}  {tier:>4d}  {reason}")
    print()


def print_per_run(runs):
    header = f"{'run':>4}  " + "  ".join(f"{n[:14]:>14}" for n in PARAM_NAMES)
    print("=" * len(header))
    print("LEARNED FINAL VALUES")
    print(header)
    print("-" * len(header))
    for r in runs:
        fp = final_params(r)
        row = f"{r['run_id']:>4d}  " + "  ".join(
            f"{scale(n, fp.get(n, float('nan'))):>14.5g}" for n in PARAM_NAMES
        )
        print(row)
    print()


def print_summary(runs):
    learned = np.array(
        [[scale(n, final_params(r).get(n, np.nan)) for n in PARAM_NAMES]
         for r in runs]
    )
    mean = np.nanmean(learned, axis=0)
    std = np.nanstd(learned, axis=0)
    cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)

    print("=" * 78)
    print(f"SUMMARY across {len(runs)} runs")
    print(f"{'parameter':>30} {'mean':>12} {'std':>12} {'CV (std/|mean|)':>18}")
    print("-" * 78)
    for i, n in enumerate(PARAM_NAMES):
        print(f"{n:>30} {mean[i]:>12.5g} {std[i]:>12.5g} {cv[i]:>18.4f}")
    print()


# Display order and pretty names for the summary table (matches the
# published table structure: diffusion coeffs, then loss/rate terms, then
# infiltration / uptake / WUE).
TABLE_ORDER = list(PARAM_NAMES)
PARAM_LABELS = {
    "surface_water_diffusion_coeff": ("Surface water diffusion",      "D_O"),
    "soil_water_diffusion_coeff":    ("Soil water diffusion",         "D_W"),
    "biomass_diffusion_coeff":       ("Plant dispersal",              "D_P"),
    "seepage_rate":                  ("Soil water loss rate",         "r_w"),
    "mortality_rate":                ("Mortality rate",               "d"),
    "infiltration_rate":             ("Max. infiltration rate",       r"\alpha"),
    "plant_uptake_rate":             ("Max. uptake rate",             r"g_{max}"),
    "water_use_efficiency":          ("Water use efficiency",         "c"),
    "infiltration_half_saturation":  ("Infiltration half-saturation", "k_2"),
    "bare_soil_infiltration":        ("Bare-soil infiltration",       "W_0"),
    "uptake_half_saturation":        ("Uptake half-saturation",       "k_1"),
}


def _fmt(value):
    """Pick decimal places based on magnitude (matches the image style:
    2 dp for |v| >= 10, 4 dp down to 0.01, scientific below that — the Rietkerk
    coefficients span five decades)."""
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.4f}"
    return f"{value:.2e}"


def print_latex_table(runs):
    """Print and save a LaTeX tabular summary: Parameter | Mean ± std | CV.

    Values are reported in physical units (m²/day, 1/day, mm, g/m²) via
    DISPLAY_SCALE.
    """
    learned = np.array(
        [[scale(n, final_params(r).get(n, np.nan)) for n in TABLE_ORDER]
         for r in runs]
    )
    mean = np.nanmean(learned, axis=0)
    std = np.nanstd(learned, axis=0)
    cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)

    lines = [
        r"\begin{tabular}{|l|c|c|}",
        r"\hline",
        r"\textbf{Parameter} & \textbf{Mean} & \textbf{CV} \\",
        r"\hline",
    ]
    for i, name in enumerate(TABLE_ORDER):
        label, sym = PARAM_LABELS[name]
        lines.append(
            f"{label} (${sym}$) & "
            f"${_fmt(mean[i])} \\pm {_fmt(std[i])}$ & "
            f"{cv[i]:.4f} \\\\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    latex = "\n".join(lines)

    # Plain-text echo of the same table for quick inspection.
    print("=" * 78)
    print(f"SUMMARY TABLE across {len(runs)} runs")
    print(f"{'Parameter':<28} {'Mean ± std':>24}  {'CV':>8}")
    print("-" * 78)
    for i, name in enumerate(TABLE_ORDER):
        label, sym = PARAM_LABELS[name]
        cell = f"{_fmt(mean[i])} ± {_fmt(std[i])}"
        print(f"{label + ' (' + sym + ')':<28} {cell:>24}  {cv[i]:>8.4f}")
    print()

    out = os.path.join(OUT_DIR, "summary_table.tex")
    with open(out, "w") as f:
        f.write(latex + "\n")
    print(f"Saved {out}\n")


def plot_parameter_trajectories(runs):
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True)
    axes = axes.flatten()
    for ax in axes[len(PARAM_NAMES):]:
        ax.axis("off")
    for i, name in enumerate(PARAM_NAMES):
        ax = axes[i]
        for r in runs:
            epochs = [e["epoch"] for e in r["parameter_history"] if name in e]
            vals = [scale(name, e[name]) for e in r["parameter_history"]
                    if name in e]
            if epochs:
                ax.plot(epochs, vals, alpha=0.5, linewidth=0.9)
        ax.set_title(PRETTY_NAMES[name], fontsize=13)
        ax.set_ylabel("Parameter value", fontsize=12)
        ax.set_yscale("log")
        if i >= len(PARAM_NAMES) - 3:
            ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    # fig.suptitle(
    #     f"Learned parameter trajectories vs epoch (real data, {len(runs)} runs)",
    #     fontsize=13,
    # )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "parameter_trajectories.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    plt.show()


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs from {RESULTS_DIR}\n")

    kept, dropped = filter_tier1(runs)
    print_dropped(dropped)

    print_per_run(kept)
    print_summary(kept)
    print_latex_table(kept)
    plot_parameter_trajectories(kept)

#%%
if __name__ == "__main__":
    main()

# %%
