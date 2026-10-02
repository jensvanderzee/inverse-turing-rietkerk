#!/usr/bin/env python3
"""
Batch Inverse PDE training script - run a range of training runs on a single GPU.

Synthetic experiment for the Rietkerk (2002) backbone (see rietkerk_model.py):
data are generated with known parameters (SYNTHETIC_TRUTH) and all eleven are
recovered from annual biomass snapshots.

Usage
-----
# Launch 3 processes in parallel, each on a different GPU:
    ./launch.sh 0  9  0   # runs 0-9  on GPU 0
    ./launch.sh 10 19 1   # runs 10-19 on GPU 1
    ./launch.sh 20 29 2   # runs 20-29 on GPU 2

# Or call the Python script directly:
    python train_invPDE_synthetic_batch.py --start 0 --end 29 --gpu 0

Each run checkpoints to <output_dir>/checkpoints/run_XX.pt (every
--checkpoint_every epochs). Rerunning the same command after a crash resumes
unfinished runs from their checkpoint and skips runs that already have a result.
"""
#%%
import argparse
import json
import math
import os
import random
import sys
import time
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

from rietkerk_model import (invRietkerk, SYNTHETIC_TRUTH, generate_weekly_precipitation,
                            draw_viable_model)

# ===========================================================================
#  0.  Reproducibility
# ===========================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
#  1.  Data Generator
# ===========================================================================
class SyntheticDataGenerator:
    def __init__(self, grid_size: Tuple[int, int] = (150, 150), device: Optional[torch.device] = None):
        self.height, self.width = grid_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ground_truth_model = invRietkerk(trainable=False, params=SYNTHETIC_TRUTH).to(self.device)

    def generate_equilibrium_state(self, equilibrium_precipitation: float = 400.0,
                                   equilibrium_years: int = 50,
                                   steps_per_week: int = 3) -> torch.Tensor:
        surface_water = torch.rand(1, 1, self.height, self.width, device=self.device)*10
        soil_water = torch.rand(1, 1, self.height, self.width, device=self.device)*10
        biomass = torch.rand(1, 1, self.height, self.width, device=self.device)*10

        uniform_weekly = generate_weekly_precipitation(
            equilibrium_precipitation, peak_week=26.0, amplitude_fraction=0.0)

        with torch.no_grad():
            for _ in range(equilibrium_years):
                surface_water, soil_water, biomass = self.ground_truth_model.simulate_year_weekly(
                    surface_water, soil_water, biomass,
                    weekly_precipitation=uniform_weekly,
                    steps_per_week=steps_per_week
                )
        return biomass.clone()

    def generate_training_data(self, weekly_precipitation_profiles: List[List[float]],
                               equilibrium_state: torch.Tensor,
                               time_series_years: int = 10,
                               noise_level: float = 0.02,
                               steps_per_week: int = 3) -> List[List[torch.Tensor]]:
        training_data = []
        for site_idx, weekly_precip in enumerate(weekly_precipitation_profiles):
            surface_water = torch.zeros(1, 1, self.height, self.width, device=self.device)
            soil_water = torch.zeros(1, 1, self.height, self.width, device=self.device)
            biomass = equilibrium_state.clone()

            site_time_series = []
            with torch.no_grad():
                for year in range(time_series_years):
                    surface_water, soil_water, biomass = self.ground_truth_model.simulate_year_weekly(
                        surface_water, soil_water, biomass,
                        weekly_precipitation=weekly_precip,
                        steps_per_week=steps_per_week
                    )
                    noise = torch.randn_like(biomass) * noise_level
                    site_time_series.append(biomass + noise)

            training_data.append(site_time_series)
        return training_data


# ===========================================================================
#  2.  Training Routine
# ===========================================================================
def train_model_adam(training_data: List[List[torch.Tensor]],
                    equilibrium_state: torch.Tensor,
                    weekly_precipitation_profiles: List[List[float]],
                    num_epochs: int = 500,
                    learning_rate: float = 0.001,
                    device: torch.device = None,
                    seed: int = None,
                    save_interval: int = 10,
                    steps_per_week: int = 3,
                    checkpoint_path: Optional[str] = None,
                    checkpoint_every: int = 50,
                    gradient_checkpointing: bool = False) -> Tuple[List[float], invRietkerk, dict, List[dict]]:
    """Fit all parameters with Adam.

    `gradient_checkpointing` recomputes each week of the rollout during the backward
    pass: same gradients for one extra forward pass, and far less memory (the
    four-site fit peaks at ~12 GB without it on CPU).

    With `checkpoint_path`, the full training state (parameters, optimiser,
    scheduler, histories) is written there every `checkpoint_every` epochs, and
    a run that finds an existing checkpoint resumes from it instead of drawing a
    new random start. The loss is deterministic, so a resumed run continues the
    same trajectory (up to float differences if it resumes on another device).
    """
    if seed is not None:
        set_seed(seed)

    # All sites share one grid, so they are simulated as a single batch (dim 0 =
    # site): one kernel launch per operation instead of one per site. The loss is
    # the same sum over sites and years of the per-site delta MSE.
    n_sites = len(weekly_precipitation_profiles)
    n_years = len(training_data[0])
    weekly_by_site = [list(week) for week in zip(*weekly_precipitation_profiles)]
    targets = [torch.cat([training_data[s][y] for s in range(n_sites)]) for y in range(n_years)]
    initial_biomass = equilibrium_state.expand(n_sites, -1, -1, -1)

    # Random start: each parameter log-uniform in rietkerk_model.INIT_RANGE (the same
    # range for every parameter, independent of SYNTHETIC_TRUTH),
    # redrawn until mean biomass stays within 0.1-10x of its initial level over the
    # training rollout (see rietkerk_model.draw_viable_model). Parameters are optimised in log space (see
    # invRietkerk), so learning_rate is a relative step size.
    session_start = time.time()
    checkpoint = None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = invRietkerk(trainable=True, reference=SYNTHETIC_TRUTH).to(device)
        model.load_state_dict(checkpoint["model"])
        model.init_draws = checkpoint["init_draws"]
        model.prior_elapsed = checkpoint["elapsed_seconds"]
        print(f"Resuming from {checkpoint_path} at epoch {checkpoint['next_epoch']}", flush=True)
    else:
        model, init_draws = draw_viable_model(
            SYNTHETIC_TRUTH, [(initial_biomass, [weekly_by_site] * n_years)],
            steps_per_week, device=device)
        model.init_draws = init_draws
        model.prior_elapsed = 0.0
        print(f"Random start accepted after {init_draws} draw(s)", flush=True)
    model.gradient_checkpointing = gradient_checkpointing

    initial_params = model.parameter_values()
    parameter_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)
    loss_history = []
    start_epoch = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initial_params = checkpoint["initial_params"]
        parameter_history = checkpoint["parameter_history"]
        loss_history = checkpoint["loss_history"]
        start_epoch = checkpoint["next_epoch"]

    def save_checkpoint(next_epoch):
        # write-then-rename, so a crash mid-write leaves the previous checkpoint intact
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "initial_params": initial_params,
                 "init_draws": model.init_draws, "parameter_history": parameter_history,
                 "loss_history": loss_history, "next_epoch": next_epoch,
                 "elapsed_seconds": model.prior_elapsed + time.time() - session_start}
        torch.save(state, checkpoint_path + ".tmp")
        os.replace(checkpoint_path + ".tmp", checkpoint_path)

    def compute_loss():
        pred_surface_water = torch.zeros_like(initial_biomass)
        pred_soil_water = torch.zeros_like(initial_biomass)
        pred_biomass = initial_biomass.clone()

        prev_pred_biomass = pred_biomass.clone()
        prev_target_biomass = pred_biomass.clone()

        total_loss = 0
        for year_idx in range(n_years):
            pred_surface_water, pred_soil_water, pred_biomass = model.simulate_year_weekly(
                pred_surface_water, pred_soil_water, pred_biomass,
                weekly_precipitation=weekly_by_site,
                steps_per_week=steps_per_week
            )
            target_biomass = targets[year_idx]
            per_site_mse = ((pred_biomass - prev_pred_biomass)
                            - (target_biomass - prev_target_biomass)).pow(2).mean(dim=(1, 2, 3))
            prev_pred_biomass = pred_biomass
            prev_target_biomass = target_biomass
            total_loss += per_site_mse.sum()
        return total_loss

    def capture_parameters(epoch):
        return {'epoch': epoch, **model.parameter_values()}

    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.clamp_parameters_()

        loss_value = loss.item()
        loss_history.append(loss_value)

        if epoch % save_interval == 0:
            parameter_history.append(capture_parameters(epoch))

        if math.isnan(loss_value):
            print(f"NaN loss encountered at epoch {epoch}", flush=True)
            break

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            param_str = "\n  ".join(f"{name}: {value:.4g}"
                                  for name, value in model.parameter_values().items())
            print(f"Epoch {epoch:4d}, Loss: {loss_value:.6f}  |  {param_str}", flush=True)

        if checkpoint_path is not None and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(epoch + 1)

    if (num_epochs - 1) % save_interval != 0:
        parameter_history.append(capture_parameters(num_epochs - 1))

    return loss_history, model, initial_params, parameter_history


# ===========================================================================
#  3.  Main Batch Loop
# ===========================================================================
#%%
class Args:
    start = 0
    end = 1
    gpu = 0
    num_epochs = 10000
    learning_rate = 0.01
    grid_size = 128
    output_dir = os.path.join("results", "synthetic_invPDE_4site_rietkerk")
    checkpoint_every = 50
    gradient_checkpointing = False
args = Args()
if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=Args.start, help="Start run ID (inclusive)")
    parser.add_argument("--end", type=int, default=Args.end, help="End run ID (inclusive)")
    parser.add_argument("--gpu", type=int, default=Args.gpu, help="GPU ID to use")
    parser.add_argument("--num_epochs", type=int, default=Args.num_epochs)
    parser.add_argument("--learning_rate", type=float, default=Args.learning_rate)
    parser.add_argument("--grid_size", type=int, default=Args.grid_size)
    parser.add_argument("--output_dir", type=str, default=Args.output_dir)
    parser.add_argument("--checkpoint_every", type=int, default=Args.checkpoint_every,
                        help="Epochs between checkpoints; a rerun resumes from the last one")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Same gradients, far less memory, one extra forward pass")
    args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(f"{args.output_dir}/models", exist_ok=True)
os.makedirs(f"{args.output_dir}/results", exist_ok=True)
os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)

print("Generating training data...", flush=True)
set_seed(42)  # Base seed so all runs share the same ground-truth

# Annual rainfall in mm/yr. The spin-up sits inside Rietkerk's Turing range
# (1.001–1.259 mm/day), so patterns form from the random start; the four sites
# span 0.71–1.29× that, i.e. 286–514 mm/yr, covering the patterned regime.
rain_multi = 400.0 / 21
steps_per_week = 2
# Data are generated on the CPU, so every run fits the same data whichever device
# trains it (CPU and GPU draw different random numbers from the same seed).
data_generator = SyntheticDataGenerator(grid_size=(args.grid_size, args.grid_size),
                                        device=torch.device("cpu"))
equilibrium_state = data_generator.generate_equilibrium_state(
    equilibrium_precipitation=21*rain_multi, equilibrium_years=100,
    steps_per_week=steps_per_week
)

n_sites = 4
annual_totals = torch.linspace(15*rain_multi, 27*rain_multi, n_sites).tolist()
peak_weeks = [26.0, 26.0, 26.0, 26.0]

weekly_precipitation_profiles = [
    generate_weekly_precipitation(annual_total=at, peak_week=pw, amplitude_fraction=0.7)
    for at, pw in zip(annual_totals, peak_weeks)
]

training_data = data_generator.generate_training_data(
    weekly_precipitation_profiles,
    equilibrium_state,
    time_series_years=10,
    noise_level=0.05,
    steps_per_week=steps_per_week
)
equilibrium_state = equilibrium_state.to(device)
training_data = [[year.to(device) for year in site] for site in training_data]


#%%
if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    import matplotlib
    matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.imshow(training_data[3][-1].squeeze().cpu())
plt.colorbar()
plt.savefig(os.path.join(args.output_dir, "training_data_site3_final.png"), dpi=100)
summary = []
# %%
for run_id in range(args.start, args.end + 1):
    result_path = os.path.join(args.output_dir, "results", f"result_{run_id:02d}.json")
    if os.path.exists(result_path):
        print(f"Run {run_id} already finished ({result_path}); skipping", flush=True)
        continue
    print(f"\n{'='*60}", flush=True)
    print(f" Starting run ID {run_id}", flush=True)
    print(f"{'='*60}", flush=True)

    run_seed = 42 + run_id
    start_time = time.time()

    loss_history, model, initial_params, parameter_history = train_model_adam(
        training_data=training_data,
        equilibrium_state=equilibrium_state,
        weekly_precipitation_profiles=weekly_precipitation_profiles,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        seed=run_seed,
        save_interval=10,
        steps_per_week=steps_per_week,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints", f"run_{run_id:02d}.pt"),
        checkpoint_every=args.checkpoint_every,
        gradient_checkpointing=args.gradient_checkpointing
    )

    elapsed = model.prior_elapsed + time.time() - start_time
    final_loss = loss_history[-1] if len(loss_history) > 0 else float('nan')

    # Save model
    model_path = os.path.join(args.output_dir, "models", f"invPDE_run_{run_id:02d}.pt")
    torch.save(model.state_dict(), model_path)

    # Save run JSON
    result_dict = {
        "run_id": run_id,
        "seed": run_seed,
        "ground_truth": SYNTHETIC_TRUTH,
        "initial_params": initial_params,
        "init_draws": model.init_draws,
        "parameter_history": parameter_history,
        "loss_history": loss_history,
        "num_epochs": len(loss_history),
        "final_loss": final_loss,
        "elapsed_seconds": elapsed
    }

    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"  Saved -> {result_path}", flush=True)

    summary.append({
        "run_id": run_id,
        "seed": run_seed,
        "num_epochs": len(loss_history),
        "final_loss": final_loss,
        "elapsed_seconds": elapsed
    })

# Write summary block
summary_path = os.path.join(args.output_dir, f"summary_{args.start:02d}-{args.end:02d}.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary written to {summary_path}", flush=True)

losses = [s["final_loss"] for s in summary if not math.isnan(s["final_loss"])]
print(f"\n{'='*60}", flush=True)
print(f"  {len(summary)} runs completed (IDs {args.start}-{args.end})", flush=True)
if losses:
    print(f"  Final loss - mean: {np.mean(losses):.6f} std: {np.std(losses):.6f}", flush=True)




# %%
