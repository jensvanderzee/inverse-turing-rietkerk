# -*- coding: utf-8 -*-
"""
Run forward simulation using the best retrained RCNN model.
Uses synthetic PDE equilibrium state as initial conditions (matching training).
Precipitation level and number of years can be changed via the settings below.

Model architecture (RCNNBaseline) and data generation (the Rietkerk ground
truth via SyntheticDataGenerator) are imported from train_rcnn_batch.py, whose
training loop only runs under `if __name__ == "__main__"`.
"""
#%%
import os
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


from train_rcnn_batch import (
    set_seed,
    SyntheticDataGenerator,
    RCNNBaseline,
    generate_weekly_precipitation,
    EQUILIBRIUM_PRECIPITATION,
    SITE_PRECIPITATION,
)


# ════════════════════════════════════════════════════════════════════════════
# SETTINGS — change these before running
# ════════════════════════════════════════════════════════════════════════════
ANNUAL_PRECIP_MM = SITE_PRECIPITATION[0]   # Annual precipitation in mm (driest training site)
NUM_YEARS        = 100       # Number of years to simulate

# Equilibrium settings (must match training in train_rcnn_batch.py)
EQUILIBRIUM_PRECIP = EQUILIBRIUM_PRECIPITATION   # 400 mm/yr, uniform spin-up
EQUILIBRIUM_YEARS  = 100    # Years of PDE spin-up for equilibrium
GRID_SIZE          = (128, 128)
DATA_SEED          = 42     # Same seed as training for reproducible initial state
# ════════════════════════════════════════════════════════════════════════════

# Paths & fixed config
RETRAIN_DIR = "results/synthetic_rcnn_4site_rietkerk"
SUMMARY_JSON = os.path.join(RETRAIN_DIR, "summary_00-09.json")
SAVE_DIR = "results/synthetic_data/rcnn_simulation_results_rietkerk"

# Model config (must match what was used in train_rcnn_batch.py)
RCNN_CONFIG = {"hidden_channels": 32, "num_layers": 1, "kernel_size": 3}


# ── Helpers ─────────────────────────────────────────────────────────────────
def find_best_run() -> dict:
    """Find the run with the lowest final training loss."""
    with open(SUMMARY_JSON) as f:
        runs = json.load(f)
    best = min(runs, key=lambda r: r["final_loss"])
    print(f"Best RCNN run: run_{best['run_id']:02d}  "
          f"(loss = {best['final_loss']:.4f}, seed = {best['seed']})")
    return best


def load_model(run_id: int, device: torch.device) -> RCNNBaseline:
    """Load a trained RCNN model from a .pt checkpoint."""
    model = RCNNBaseline(**RCNN_CONFIG).to(device)
    pt_path = os.path.join(RETRAIN_DIR, f"run_{run_id:02d}.pt")
    checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded {pt_path}")
    return model


def generate_initial_conditions(device: torch.device) -> torch.Tensor:
    """Generate synthetic equilibrium biomass using the ground-truth PDE model.

    Uses the same seed, grid size, and equilibrium settings as training
    so the initial state is identical to what the RCNN was trained on.
    """
    print(f"Generating synthetic equilibrium state ...")
    print(f"  Grid: {GRID_SIZE[0]}x{GRID_SIZE[1]}, "
          f"precip: {EQUILIBRIUM_PRECIP}, years: {EQUILIBRIUM_YEARS}, seed: {DATA_SEED}")

    set_seed(DATA_SEED)
    data_gen = SyntheticDataGenerator(grid_size=GRID_SIZE, device=device)
    with torch.no_grad():
        initial_biomass = data_gen.generate_equilibrium_state(
            equilibrium_precipitation=EQUILIBRIUM_PRECIP,
            equilibrium_years=EQUILIBRIUM_YEARS,
        )

    print(f"  Shape: {tuple(initial_biomass.shape)}, "
          f"range: {initial_biomass.min().item():.1f} – {initial_biomass.max().item():.1f}")
    return initial_biomass


def simulate(
    model: RCNNBaseline,
    initial_biomass: torch.Tensor,
    annual_precip_mm: float,
    num_years: int,
    device: torch.device,
) -> dict:
    """
    Run the RCNN forward for *num_years* years at a constant precipitation.
    Each year: 52 weekly forward steps through the model.

    Returns dict with biomass snapshots (index 0 = initial state).
    """
    weekly_precip = generate_weekly_precipitation(annual_precip_mm)

    b = initial_biomass.clone()
    hidden = None

    snapshots = {"biomass": [b.squeeze().cpu().numpy()]}

    print(f"\nSimulating {num_years} years at {annual_precip_mm:.1f} mm/yr ...")
    with torch.no_grad():
        for yr in range(1, num_years + 1):
            b, hidden = model.simulate_year_weekly(b, weekly_precip, hidden)
            b = b.detach()
            hidden = [h.detach() for h in hidden]

            snapshots["biomass"].append(b.squeeze().cpu().numpy())

            if yr % max(1, num_years // 10) == 0 or yr == num_years:
                print(f"  Year {yr:>4d}  |  biomass mean={b.mean().item():.2f}  "
                      f"min={b.min().item():.2f}  max={b.max().item():.2f}")

    return snapshots


# ── Visualisation ───────────────────────────────────────────────────────────
def plot_results(snapshots: dict, annual_precip_mm: float, num_years: int,
                 run_id: int, save_dir: str):
    """Create and save summary figures."""
    biomass = np.array(snapshots["biomass"])
    n_frames = biomass.shape[0]

    # ── 1. Time series of spatial statistics ────────────────────────────────
    means = [b.mean() for b in biomass]
    stds = [b.std() for b in biomass]
    mins = [b.min() for b in biomass]
    maxs = [b.max() for b in biomass]
    years = list(range(n_frames))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, means, "o-", label="mean")
    ax.fill_between(
        years,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25,
        label="mean +/- 1 std",
    )
    ax.plot(years, mins, "v--", ms=4, label="min")
    ax.plot(years, maxs, "^--", ms=4, label="max")
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Biomass")
    ax.set_title(f"RCNN biomass evolution — run {run_id:02d}, {annual_precip_mm:.0f} mm/yr")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "biomass_timeseries.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved biomass_timeseries.png")

    # ── 2. Spatial snapshots at selected time points ────────────────────────
    if n_frames <= 10:
        indices = list(range(n_frames))
    else:
        indices = sorted(set(
            [0]
            + list(np.linspace(0, n_frames - 1, 10, dtype=int))
            + [n_frames - 1]
        ))

    ncols = min(5, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    axes = np.atleast_2d(axes)

    vmax = max(biomass[0].max(), biomass[-1].max(), 1.0)
    norm = Normalize(vmin=0, vmax=vmax)

    for ax_idx, frame_idx in enumerate(indices):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r, c]
        im = ax.imshow(biomass[frame_idx], cmap="RdYlGn", norm=norm)
        ax.set_title(f"Year {frame_idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax_idx in range(len(indices), nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"RCNN biomass snapshots — run {run_id:02d}, {annual_precip_mm:.0f} mm/yr",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "biomass_snapshots.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved biomass_snapshots.png")

    # ── 3. Initial vs final comparison ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, idx) in zip(axes, [("Initial (year 0)", 0), (f"Final (year {n_frames - 1})", -1)]):
        im = ax.imshow(biomass[idx], cmap="RdYlGn", norm=norm)
        ax.set_title(f"{label}\nmean={biomass[idx].mean():.2f}, "
                     f"min={biomass[idx].min():.2f}, max={biomass[idx].max():.2f}",
                     fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"RCNN initial vs final — run {run_id:02d}, {annual_precip_mm:.0f} mm/yr, {num_years} years",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "biomass_initial_vs_final.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved biomass_initial_vs_final.png")


# ── Main ────────────────────────────────────────────────────────────────────
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

run_tag = f"precip{ANNUAL_PRECIP_MM:.0f}_years{NUM_YEARS}"
save_dir = os.path.join(SAVE_DIR, run_tag)
os.makedirs(save_dir, exist_ok=True)

# 1. Find and load best model
best_run = find_best_run()
run_id = best_run["run_id"]
model = load_model(run_id, device)

# 2. Generate synthetic equilibrium initial conditions
initial_biomass = generate_initial_conditions(device)

# 3. Simulate
snapshots = simulate(model, initial_biomass, ANNUAL_PRECIP_MM, NUM_YEARS, device)

# 4. Save config + numerical results
config = {
    "run_id": run_id,
    "annual_precip_mm": ANNUAL_PRECIP_MM,
    "num_years": NUM_YEARS,
    "rcnn_config": RCNN_CONFIG,
    "initial_conditions": "synthetic_equilibrium",
    "equilibrium_precipitation": EQUILIBRIUM_PRECIP,
    "equilibrium_years": EQUILIBRIUM_YEARS,
    "grid_size": list(GRID_SIZE),
    "data_seed": DATA_SEED,
    "final_training_loss": best_run["final_loss"],
}
with open(os.path.join(save_dir, "simulation_config.json"), "w") as f:
    json.dump(config, f, indent=2)

biomass = np.array(snapshots["biomass"])
stats_df = pd.DataFrame({
    "year": list(range(biomass.shape[0])),
    "biomass_mean": [b.mean() for b in biomass],
    "biomass_std": [b.std() for b in biomass],
    "biomass_min": [b.min() for b in biomass],
    "biomass_max": [b.max() for b in biomass],
})
stats_df.to_csv(os.path.join(save_dir, "biomass_stats.csv"), index=False)

# 5. Plots
plot_results(snapshots, ANNUAL_PRECIP_MM, NUM_YEARS, run_id, save_dir)

print(f"\nAll outputs saved to {save_dir}/")

# %%
