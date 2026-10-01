# -*- coding: utf-8 -*-
"""
One-at-a-time (OAT) sensitivity analysis for the Rietkerk PDE model
(rietkerk_model.py), under the protocol of the synthetic experiments: spin-up
under uniform rain, then ten years of seasonal weekly forcing per site, sampled
once a year.

For each parameter:
  1. Run with ground-truth values → baseline biomass fields
  2. Perturb *one* parameter by a set of relative deltas (e.g. ±5 %, ±10 %, ±20 %)
  3. Re-run the model and compare the biomass output pixel-by-pixel to baseline
  4. Quantify sensitivity as the pixel-wise normalised RMSE

Outputs (saved to results/synthetic/sensitivity_analysis_rietkerk/):
  - sensitivity_summary.csv            per-parameter sensitivity indices
  - sensitivity_bar.png                bar chart ranking parameters
  - sensitivity_curves.png             sensitivity vs perturbation magnitude
  - elasticity.png                     dimensionless elasticity ranking
  - pixel_sensitivity_maps.png         per-pixel RMSE maps (one panel per parameter)
  - biomass_timeseries_<param>.png     overlay of perturbed vs baseline time series
  - spatial_difference_<param>.png     maps of biomass difference at final time step
"""
# %%
import os
import random
import warnings
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from rietkerk_model import invRietkerk, SYNTHETIC_TRUTH, generate_weekly_precipitation

# Synthetic-experiment protocol (train_invPDE_synthetic_batch.py)
GRID_SIZE = 128
STEPS_PER_WEEK = 2
SPINUP_PRECIPITATION = 400.0      # mm/yr, uniform over the year
SPINUP_YEARS = 100


# ── Sensitivity-analysis helpers ───────────────────────────────────────────

GROUND_TRUTH: Dict[str, float] = dict(SYNTHETIC_TRUTH)

PARAM_LABELS = {
    "infiltration_rate": "Infiltration (α)",
    "seepage_rate": "Soil water loss (r_w)",
    "plant_uptake_rate": "Plant uptake (g_max)",
    "mortality_rate": "Mortality (d)",
    "water_use_efficiency": "WUE (c)",
    "infiltration_half_saturation": "Infiltr. half-sat. (k2)",
    "bare_soil_infiltration": "Bare-soil infiltr. (W0)",
    "uptake_half_saturation": "Uptake half-sat. (k1)",
    "surface_water_diffusion_coeff": "Surf. water diff.",
    "soil_water_diffusion_coeff": "Soil water diff.",
    "biomass_diffusion_coeff": "Biomass diff.",
}

# Perturbation magnitudes (relative)
PERTURBATIONS = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_model(device: torch.device, overrides: Dict[str, float] = None):
    """Create a fresh model with ground-truth params, optionally overriding some."""
    params = {**GROUND_TRUTH, **(overrides or {})}
    return invRietkerk(params=params, reference=GROUND_TRUTH).to(device)


@torch.no_grad()
def generate_equilibrium(model: invRietkerk, device: torch.device,
                         precip: float = SPINUP_PRECIPITATION, years: int = SPINUP_YEARS):
    """Spin-up the model to reach spatial equilibrium (uniform rain, mm/yr)."""
    sw = torch.rand(1, 1, GRID_SIZE, GRID_SIZE, device=device) * 10
    slw = torch.rand(1, 1, GRID_SIZE, GRID_SIZE, device=device) * 10
    bio = torch.rand(1, 1, GRID_SIZE, GRID_SIZE, device=device) * 10

    uniform_weekly = generate_weekly_precipitation(precip, amplitude_fraction=0.0)
    for _ in range(years):
        sw, slw, bio = model.simulate_year_weekly(sw, slw, bio, uniform_weekly,
                                                  steps_per_week=STEPS_PER_WEEK)

    return bio.clone()


@torch.no_grad()
def run_paired_pixelwise(
    baseline_model: invRietkerk,
    perturbed_model: invRietkerk,
    equilibrium_state: torch.Tensor,
    precipitation_values: List[float],
    time_series_years: int,
    device: torch.device,
) -> Dict[str, object]:
    """
    Run baseline and perturbed models in lockstep, accumulating pixel-wise
    squared differences on the GPU without storing all intermediate fields.

    Returns dict with:
      - 'nrmse'              : scalar — pixel-wise NRMSE over all pixels,
                               years and sites
      - 'pixel_sensitivity'  : (H, W) array — per-pixel RMSE averaged
                               across years and sites
      - 'mean_biomass_base'  : list — spatial-mean baseline biomass per
                               year (site 0, for time-series plots)
      - 'mean_biomass_pert'  : list — same for the perturbed run
      - 'final_biomass_base' : (H, W) — baseline biomass in the last year
                               (site 0)
      - 'final_biomass_pert' : (H, W) — perturbed biomass in the last year
                               (site 0)

    `precipitation_values` are annual totals (mm/yr), each delivered with the
    seasonal weekly profile of the synthetic experiments.
    """
    H, W = equilibrium_state.shape[-2:]

    # Accumulators for pixel-wise comparison
    pixel_sse_sum = torch.zeros(H, W, device=device)
    pixel_base_abs_sum = torch.zeros(H, W, device=device)
    n_samples_total = 0

    # Time-series and spatial caches (site 0 only)
    mean_base_ts: List[float] = []
    mean_pert_ts: List[float] = []
    final_base = None
    final_pert = None

    for site_idx, precip in enumerate(precipitation_values):
        weekly = generate_weekly_precipitation(precip, amplitude_fraction=0.7)
        # Identical initial conditions for both runs
        sw_b  = torch.zeros(1, 1, H, W, device=device)
        slw_b = torch.zeros(1, 1, H, W, device=device)
        bio_b = equilibrium_state.clone()

        sw_p  = torch.zeros(1, 1, H, W, device=device)
        slw_p = torch.zeros(1, 1, H, W, device=device)
        bio_p = equilibrium_state.clone()

        site_pixel_sse = torch.zeros(H, W, device=device)
        site_pixel_base = torch.zeros(H, W, device=device)

        for year in range(time_series_years):
            sw_b, slw_b, bio_b = baseline_model.simulate_year_weekly(
                sw_b, slw_b, bio_b, weekly, steps_per_week=STEPS_PER_WEEK)
            sw_p, slw_p, bio_p = perturbed_model.simulate_year_weekly(
                sw_p, slw_p, bio_p, weekly, steps_per_week=STEPS_PER_WEEK)

            # One sample per simulated year
            diff = (bio_p - bio_b).squeeze()
            site_pixel_sse += diff ** 2
            site_pixel_base += bio_b.squeeze().abs()
            n_samples_total += 1

            # Track mean-biomass time series for site 0
            if site_idx == 0:
                mean_base_ts.append(bio_b.mean().item())
                mean_pert_ts.append(bio_p.mean().item())

        pixel_sse_sum += site_pixel_sse
        pixel_base_abs_sum += site_pixel_base

        # Cache final fields for site 0
        if site_idx == 0:
            final_base = bio_b.squeeze().cpu().numpy()
            final_pert = bio_p.squeeze().cpu().numpy()

    # ── Aggregate ──────────────────────────────────────────────────────────
    pixel_rmse = torch.sqrt(pixel_sse_sum / n_samples_total).cpu().numpy()

    total_mse = (pixel_sse_sum.sum() / n_samples_total).item() / (H * W)
    total_rmse = np.sqrt(total_mse)
    mean_abs_base = (pixel_base_abs_sum.sum()
                     / (n_samples_total * H * W)).item()
    nrmse = total_rmse / mean_abs_base if mean_abs_base > 1e-12 else 0.0

    return {
        "nrmse": nrmse,
        "pixel_sensitivity": pixel_rmse,
        "mean_biomass_base": mean_base_ts,
        "mean_biomass_pert": mean_pert_ts,
        "final_biomass_base": final_base,
        "final_biomass_pert": final_pert,
    }


# ── Plotting helpers ───────────────────────────────────────────────────────

def plot_sensitivity_bar(results_df: pd.DataFrame, save_path: str):
    """Rank parameters by sensitivity at the reference perturbation (+10 %)."""
    ref = results_df[results_df["perturbation"] == 0.10].copy()
    if ref.empty:
        ref = results_df.groupby("parameter").agg({"nrmse": "max"}).reset_index()
    ref = ref.sort_values("nrmse", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [PARAM_LABELS.get(p, p) for p in ref["parameter"]]
    bars = ax.barh(labels, ref["nrmse"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Pixel-wise Normalised RMSE (+10 % perturbation)", fontsize=12)
    ax.set_title("Parameter Sensitivity Ranking", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, ref["nrmse"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sensitivity_bar.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_sensitivity_curves(results_df: pd.DataFrame, save_path: str):
    """Pixel-wise NRMSE vs perturbation magnitude for every parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    params = results_df["parameter"].unique()
    for i, param in enumerate(params):
        sub = results_df[results_df["parameter"] == param].sort_values("perturbation")
        ax.plot(sub["perturbation"] * 100, sub["nrmse"],
                marker="o", label=PARAM_LABELS.get(param, param),
                color=cmap(i / len(params)), linewidth=1.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Perturbation (%)", fontsize=12)
    ax.set_ylabel("Pixel-wise Normalised RMSE", fontsize=12)
    ax.set_title("Sensitivity Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sensitivity_curves.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


def plot_biomass_timeseries(param_name: str,
                            baseline_means: List[float],
                            perturbed_runs: Dict[float, List[float]],
                            save_path: str):
    """Overlay baseline and perturbed mean-biomass time series for one parameter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(len(baseline_means))
    ax.plot(t, baseline_means, "k-", linewidth=2, label="Baseline")
    cmap = plt.cm.coolwarm
    deltas = sorted(perturbed_runs.keys())
    norm = plt.Normalize(vmin=min(deltas), vmax=max(deltas))
    for delta in deltas:
        ax.plot(t, perturbed_runs[delta], linewidth=1.2,
                color=cmap(norm(delta)),
                label=f"{delta:+.0%}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean biomass")
    ax.set_title(f"Biomass response to perturbation of "
                 f"{PARAM_LABELS.get(param_name, param_name)}",
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"biomass_timeseries_{param_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_spatial_difference(param_name: str,
                            baseline_field: np.ndarray,
                            perturbed_fields: Dict[float, np.ndarray],
                            save_path: str):
    """Show spatial maps of biomass difference at final time step."""
    deltas = sorted(perturbed_fields.keys())
    n = len(deltas)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    vmax = max(np.abs(perturbed_fields[d] - baseline_field).max() for d in deltas)
    vmax = max(vmax, 1e-6)

    for ax, delta in zip(axes, deltas):
        diff = perturbed_fields[delta] - baseline_field
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{delta:+.0%}", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Biomass difference (perturbed − baseline)\n"
                 f"Parameter: {PARAM_LABELS.get(param_name, param_name)}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"spatial_difference_{param_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_pixel_sensitivity_maps(pixel_maps: Dict[str, np.ndarray],
                                 save_path: str):
    """Grid of per-pixel RMSE maps (one per parameter, at +10 % perturbation)."""
    params = list(pixel_maps.keys())
    n = len(params)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        im = ax.imshow(pixel_maps[param], cmap="inferno")
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=12,
                     fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-pixel sensitivity (RMSE over time)\n+10 % perturbation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "pixel_sensitivity_maps.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_elasticity(results_df: pd.DataFrame, save_path: str):
    """
    Elasticity = (Δoutput / output) / (Δparameter / parameter).
    Estimated from small perturbations (±5 %).  A dimensionless measure
    of proportional sensitivity that is comparable across parameters.
    """
    small = results_df[results_df["perturbation"].abs() <= 0.05].copy()
    if small.empty:
        return

    small["elasticity"] = small["nrmse"] / small["perturbation"].abs()
    elas = small.groupby("parameter")["elasticity"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [PARAM_LABELS.get(p, p) for p in elas.index]
    ax.barh(labels, elas.values, color="darkorange", edgecolor="white")
    ax.set_xlabel("Elasticity (dimensionless)", fontsize=12)
    ax.set_title("Parameter Elasticity\n"
                 "(proportional output change per proportional input change)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, (lbl, val) in enumerate(zip(labels, elas.values)):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "elasticity.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = os.path.join("results", "synthetic", "sensitivity_analysis_rietkerk")
    os.makedirs(save_dir, exist_ok=True)

    # -- 1. Generate equilibrium with ground-truth model ────────────────────
    print("Generating equilibrium state …")
    gt_model = build_model(device)
    equilibrium = generate_equilibrium(gt_model, device)

    # Simulation settings: the four synthetic sites (mm/yr), ten years each
    precipitation_values = torch.linspace(15, 27, 4).mul(SPINUP_PRECIPITATION / 21).tolist()
    time_series_years = 10

    # -- 2. Perturbed runs (paired pixel-wise) ──────────────────────────────
    rows = []
    # Caches for per-parameter plots (site 0)
    baseline_mean_ts: List[float] = []       # set once from first run
    ts_cache: Dict[str, Dict[float, List[float]]] = {}
    field_cache_base: Dict[str, np.ndarray] = {}
    field_cache_pert: Dict[str, Dict[float, np.ndarray]] = {}
    pixel_maps_10pct: Dict[str, np.ndarray] = {}

    for param_name, gt_val in GROUND_TRUTH.items():
        print(f"  Perturbing {param_name} (GT = {gt_val}) …")
        ts_cache[param_name] = {}
        field_cache_pert[param_name] = {}

        for delta in PERTURBATIONS:
            new_val = gt_val * (1.0 + delta)
            perturbed_model = build_model(device,
                                          overrides={param_name: new_val})

            pw = run_paired_pixelwise(
                gt_model, perturbed_model, equilibrium,
                precipitation_values, time_series_years, device,
            )

            rows.append({
                "parameter": param_name,
                "gt_value": gt_val,
                "perturbation": delta,
                "new_value": new_val,
                "nrmse": pw["nrmse"],
            })

            # Cache time series and fields for plots (site 0)
            ts_cache[param_name][delta] = pw["mean_biomass_pert"]
            field_cache_pert[param_name][delta] = pw["final_biomass_pert"]

            # Baseline is the same for every delta; grab it once
            if not baseline_mean_ts:
                baseline_mean_ts = pw["mean_biomass_base"]
            if param_name not in field_cache_base:
                field_cache_base[param_name] = pw["final_biomass_base"]

            if abs(delta - 0.10) < 1e-9:
                pixel_maps_10pct[param_name] = pw["pixel_sensitivity"]

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(save_dir, "sensitivity_summary.csv"),
                      index=False)
    print(f"\nSaved sensitivity_summary.csv  ({len(results_df)} runs)")

    # -- 3. Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SENSITIVITY SUMMARY (pixel-wise NRMSE at ±10 % perturbation)")
    print(f"{'='*72}")
    ref = results_df[results_df["perturbation"].isin([0.10, -0.10])]
    summary = ref.groupby("parameter")["nrmse"].mean().sort_values(ascending=False)
    for p, v in summary.items():
        print(f"  {PARAM_LABELS.get(p, p):>20s}   NRMSE = {v:.6f}")

    # -- 4. Plots ───────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_sensitivity_bar(results_df, save_dir)
    plot_sensitivity_curves(results_df, save_dir)
    plot_elasticity(results_df, save_dir)

    if pixel_maps_10pct:
        plot_pixel_sensitivity_maps(pixel_maps_10pct, save_dir)

    for param_name in GROUND_TRUTH:
        plot_biomass_timeseries(param_name, baseline_mean_ts,
                                ts_cache[param_name], save_dir)
        big_deltas = {d: field_cache_pert[param_name][d]
                      for d in [-0.20, -0.10, 0.10, 0.20]
                      if d in field_cache_pert[param_name]}
        if big_deltas:
            plot_spatial_difference(param_name, field_cache_base[param_name],
                                   big_deltas, save_dir)

    print(f"\nAll outputs saved to:  {save_dir}/")


if __name__ == "__main__":
    main()
