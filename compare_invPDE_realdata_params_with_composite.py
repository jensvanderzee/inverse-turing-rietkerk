
#%%
import csv
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from rietkerk_model import (PARAM_NAMES, PRETTY_NAMES, REALDATA_REFERENCE,
                            NDVI_TO_RIETKERK_GRAMS, composite_value,
                            to_physical_units, degenerate_parameters)

RESULTS_DIR      = os.path.join("results", "real_data_rietkerk", "models", "parameters")
TEST_METRICS_CSV = os.path.join("results", "real_data_rietkerk", "test_results", "test_metrics.csv")
OUT_DIR = os.path.join("results", "real_data_rietkerk", "param_comparison")
DATA_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# Training subsites used for the real-data models (same as in
# realdata_train_invPDE.py).
TRAINING_SUBSITES = ["b", "i", "c", "e"]
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0

# Tier 1 filter thresholds (applied to raw, un-scaled values).
LENGTH_FRAC = 0.9      # drop runs with fewer than 90% of the max snapshot count
# Runs with any final param NaN/inf or on its clamp bound are dropped too
# (rietkerk_model.degenerate_parameters).

# Unit conversion for display and reporting: Rietkerk's physical units (m²/day
# for diffusion, g/m² for biomass, see rietkerk_model.to_physical_units) instead
# of pixel²/day and NDVI x 1500. The filter above still operates on raw values.
_PHYSICAL = to_physical_units(REALDATA_REFERENCE, ndvi_to_biomass_multiplier=NDVI_TO_BIOMASS_MULTIPLIER)
DISPLAY_SCALE = {name: _PHYSICAL[name] / REALDATA_REFERENCE[name] for name in PARAM_NAMES}


def scale(name, value):
    """Return value in display units (raw * DISPLAY_SCALE[name] if present)."""
    return value * DISPLAY_SCALE.get(name, 1.0)


def compute_mean_training_biomass(data_dir=DATA_DIR,
                                  subsites=TRAINING_SUBSITES,
                                  multiplier=NDVI_TO_BIOMASS_MULTIPLIER):
    """Mean biomass across all NDVI images used to train the real-data models.

    NDVI = (NIR - Red)/(NIR + Red); biomass = clip(max(NDVI, 0) * multiplier,
    0, multiplier) — identical to realdata_train_invPDE._load_satellite_image.
    """
    per_image_means = []
    for site in subsites:
        ndvi_dir = os.path.join(data_dir, f"subsite_{site}", f"subsite_{site}_ndvi")
        if not os.path.isdir(ndvi_dir):
            print(f"Warning: NDVI directory not found for subsite {site}: {ndvi_dir}")
            continue
        for fname in sorted(f for f in os.listdir(ndvi_dir)
                            if f.lower().endswith(".tif")):
            fpath = os.path.join(ndvi_dir, fname)
            with rasterio.open(fpath) as src:
                red = src.read(1).astype(np.float32)
                nir = src.read(2).astype(np.float32)
                denom = nir + red
                ndvi = np.zeros_like(denom)
                valid = denom > 0
                ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
                biomass = np.clip(np.maximum(ndvi, 0) * multiplier, 0, multiplier)
                per_image_means.append(float(biomass.mean()))
    if not per_image_means:
        raise RuntimeError(
            f"No NDVI images found under {data_dir} for subsites {subsites}"
        )
    B = float(np.mean(per_image_means))
    print(
        f"Mean training biomass over {len(per_image_means)} images "
        f"(subsites {subsites}, multiplier={multiplier}): B = {B:.6f}"
    )
    return B


def compute_composite(fp, B):
    """Rain use efficiency (g/m² of biomass per mm of rain) at mean biomass B (in
    data units): rietkerk_model.composite_value evaluated in physical units."""
    B_grams = B * NDVI_TO_RIETKERK_GRAMS / NDVI_TO_BIOMASS_MULTIPLIER
    return composite_value(
        to_physical_units(fp, ndvi_to_biomass_multiplier=NDVI_TO_BIOMASS_MULTIPLIER), B_grams)


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


def print_per_run(runs, B=None):
    cols = list(PARAM_NAMES)
    if B is not None:
        cols = cols + ["composite"]
    header = f"{'run':>4}  " + "  ".join(f"{n[:14]:>14}" for n in cols)
    print("=" * len(header))
    print("LEARNED FINAL VALUES")
    print(header)
    print("-" * len(header))
    for r in runs:
        fp = final_params(r)
        values = [scale(n, fp.get(n, float("nan"))) for n in PARAM_NAMES]
        if B is not None:
            values.append(compute_composite(fp, B))
        row = f"{r['run_id']:>4d}  " + "  ".join(f"{v:>14.5g}" for v in values)
        print(row)
    print()


def print_summary(runs, B=None):
    cols = list(PARAM_NAMES)
    rows = [[scale(n, final_params(r).get(n, np.nan)) for n in PARAM_NAMES]
            for r in runs]
    if B is not None:
        cols = cols + ["composite"]
        for i, r in enumerate(runs):
            rows[i].append(compute_composite(final_params(r), B))

    learned = np.array(rows)
    mean = np.nanmean(learned, axis=0)
    std = np.nanstd(learned, axis=0)
    cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)

    print("=" * 78)
    print(f"SUMMARY across {len(runs)} runs")
    print(f"{'parameter':>30} {'mean':>12} {'std':>12} {'CV (std/|mean|)':>18}")
    print("-" * 78)
    for i, n in enumerate(cols):
        print(f"{n:>30} {mean[i]:>12.5g} {std[i]:>12.5g} {cv[i]:>18.4f}")
    print()


# Display order and pretty names for the summary table (diffusion coeffs, then
# loss/rate terms, then infiltration / uptake / WUE and the saturation constants,
# with the derived composite appended).
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
PARAM_UNITS = {
    "surface_water_diffusion_coeff": r"$m^{2}/d$",
    "soil_water_diffusion_coeff":    r"$m^{2}/d$",
    "biomass_diffusion_coeff":       r"$m^{2}/d$",
    "seepage_rate":                  r"$1/d$",
    "mortality_rate":                r"$1/d$",
    "infiltration_rate":             r"$1/d$",
    "plant_uptake_rate":             r"$mm\, m^{2}/(g\, d)$",
    "water_use_efficiency":          r"$g/(mm\, m^{2})$",
    "infiltration_half_saturation":  r"$g/m^{2}$",
    "bare_soil_infiltration":        r"$1$",
    "uptake_half_saturation":        r"$mm$",
}
COMPOSITE_LABEL = (
    "Composite",
    r"\frac{g_{max} B / k_1}{r_w + g_{max} B / k_1}\,c",
)


def _fmt(value):
    """Pick decimal places based on magnitude (matches the image style:
    2 dp for |v| >= 10, 4 dp down to 0.01, scientific below that — the Rietkerk
    coefficients span five decades)."""
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.4f}"
    return f"{value:.2e}"


def print_latex_table(runs, B=None):
    """Print and save a LaTeX tabular summary: Parameter | Mean ± std | CV.

    Values for the base parameters are reported in physical units (m²/day,
    1/day, mm, g/m²) via DISPLAY_SCALE. The composite row, if B is provided, is
    computed per-run from the learned parameters and the training-time mean
    biomass B, in g/m² of biomass per mm of rain.
    """
    base_rows = np.array(
        [[scale(n, final_params(r).get(n, np.nan)) for n in TABLE_ORDER]
         for r in runs]
    )
    base_mean = np.nanmean(base_rows, axis=0)
    base_std = np.nanstd(base_rows, axis=0)
    base_cv = np.where(np.abs(base_mean) > 0, base_std / np.abs(base_mean), np.nan)

    if B is not None:
        comp_values = np.array(
            [compute_composite(final_params(r), B) for r in runs]
        )
        comp_mean = float(np.nanmean(comp_values))
        comp_std = float(np.nanstd(comp_values))
        comp_cv = (comp_std / abs(comp_mean)) if abs(comp_mean) > 0 else float("nan")

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
            f"${_fmt(base_mean[i])} \\pm {_fmt(base_std[i])}$ & "
            f"{base_cv[i]:.4f} \\\\"
        )
        lines.append(r"\hline")
    if B is not None:
        label, sym = COMPOSITE_LABEL
        lines.append(
            f"{label} (${sym}$) & "
            f"${_fmt(comp_mean)} \\pm {_fmt(comp_std)}$ & "
            f"{comp_cv:.4f} \\\\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    latex = "\n".join(lines)

    # Plain-text echo of the same table for quick inspection.
    print("=" * 78)
    print(f"SUMMARY TABLE across {len(runs)} runs")
    if B is not None:
        print(f"(composite uses mean training biomass B = {B:.4f})")
    print(f"{'Parameter':<28} {'Mean ± std':>24}  {'CV':>8}")
    print("-" * 78)
    for i, name in enumerate(TABLE_ORDER):
        label, sym = PARAM_LABELS[name]
        cell = f"{_fmt(base_mean[i])} ± {_fmt(base_std[i])}"
        print(f"{label + ' (' + sym + ')':<28} {cell:>24}  {base_cv[i]:>8.4f}")
    if B is not None:
        label, _sym = COMPOSITE_LABEL
        cell = f"{_fmt(comp_mean)} ± {_fmt(comp_std)}"
        print(f"{label:<28} {cell:>24}  {comp_cv:>8.4f}")
    print()

    out = os.path.join(OUT_DIR, "summary_table_with_composite.tex")
    with open(out, "w") as f:
        f.write(latex + "\n")
    print(f"Saved {out}\n")


def load_test_mse():
    """Return {model_id: mean_mse} averaged across test sites from test_metrics.csv."""
    mse_map = {}
    try:
        site_totals, site_counts = {}, {}
        with open(TEST_METRICS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                mid = int(row["model_id"])
                mse_str = row.get("mse", "").strip()
                if mse_str:
                    site_totals[mid] = site_totals.get(mid, 0.0) + float(mse_str)
                    site_counts[mid]  = site_counts.get(mid, 0) + 1
        for mid, total in site_totals.items():
            if site_counts[mid] > 0:
                mse_map[mid] = total / site_counts[mid]
    except FileNotFoundError:
        print(f"Warning: {TEST_METRICS_CSV} not found; test MSE column will be NaN.")
    return mse_map


def _fmt_allmodels(value):
    """2 dp for |v| >= 10, 3 dp down to 0.01, scientific below that."""
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def _cell_param(fp, name, degenerate):
    """Format one parameter cell: a dash for a value on its clamp bound (failed
    run), else the value in physical units."""
    if name in degenerate:
        return r"\textemdash"
    return _fmt_allmodels(scale(name, fp.get(name, float("nan"))))


def print_latex_all_models_table(runs, B=None):
    """Print and save a per-model LaTeX table with all runs, sorted by test MSE.

    One column per learned parameter (TABLE_ORDER, physical units) plus an extra
    Rain use efficiency (phi) column computed as the composite
        phi = (g_max B / k1) / (r_w + g_max B / k1) * c
    at mean training biomass B.  The phi column is omitted when B is None.
    """
    mse_map = load_test_mse()

    rows = []
    for r in runs:
        fp  = final_params(r)
        mse = mse_map.get(r["run_id"], float("nan"))
        comp = compute_composite(fp, B) if B is not None else float("nan")
        rows.append((mse, r["run_id"], fp, comp))
    rows.sort(key=lambda x: x[0])

    has_phi = B is not None
    ncols   = 2 + len(TABLE_ORDER) + (1 if has_phi else 0)
    col_spec = "|".join(["c"] * ncols)

    def row_(*cells):
        return "    " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[htbp]",
        (r"    \caption{Learned Rietkerk parameters of all training runs using "
         r"satellite data, including their MSE on a test set of three locations. "
         r"A dash marks a parameter that ran onto its bound, i.e. a failed "
         r"training run.}"),
        r"    \label{tab:all_models_realdata}",
        rf"    \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    header = [r"\textbf{Model ID}", r"\textbf{Test MSE}"]
    header += [rf"\textbf{{{PARAM_LABELS[n][0]} (${PARAM_LABELS[n][1]}$)}}" for n in TABLE_ORDER]
    units = [r"", r""] + [PARAM_UNITS[n] for n in TABLE_ORDER]
    if has_phi:
        header.append(r"\textbf{Rain use efficiency ($\phi$)}")
        units.append(r"$g/(m^{2}\, mm)$")
    lines += [row_(*header), r"    \hline", row_(*units), r"    \hline", r"    ", r"    \midrule"]

    for mse, run_id, fp, comp in rows:
        degenerate = set(degenerate_parameters(fp, REALDATA_REFERENCE))
        cells = [str(run_id), f"{mse:.2f}"]
        for name in TABLE_ORDER:
            cells.append(_cell_param(fp, name, degenerate))
        if has_phi:
            if degenerate or not np.isfinite(comp):
                cells.append(r"\textemdash")
            else:
                cells.append(_fmt_allmodels(comp))
        lines.append(row_(*cells))

    lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    out = os.path.join(OUT_DIR, "all_models_table.tex")
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
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "parameter_trajectories.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    plt.show()


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs from {RESULTS_DIR}\n")

    try:
        B = compute_mean_training_biomass()
    except Exception as e:
        print(f"Could not compute mean training biomass ({e}); "
              f"composite row will be skipped.")
        B = None

    kept, dropped = filter_tier1(runs)
    print_dropped(dropped)

    print_per_run(kept, B=B)
    print_summary(kept, B=B)
    print_latex_table(kept, B=B)
    print_latex_all_models_table(kept, B=B)
    plot_parameter_trajectories(kept)

#%%
if __name__ == "__main__":
    main()

# %%
