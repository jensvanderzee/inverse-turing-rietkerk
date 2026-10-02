#!/usr/bin/env python3
"""
Batch Inverse PDE training script - run a range of training runs on a single GPU.

Single-site synthetic experiment for the Rietkerk (2002) backbone (see
rietkerk_model.py). Same ground truth as the four-site experiment.

Usage
-----
# Launch 3 processes in parallel, each on a different GPU:
    ./launch.sh 0  9  0   # runs 0-9  on GPU 0
    ./launch.sh 10 19 1   # runs 10-19 on GPU 1
    ./launch.sh 20 29 2   # runs 20-29 on GPU 2

# Or call the Python script directly:
    python train_invPDE_synthetic_batch_1site.py --start 0 --end 29 --gpu 0
"""

import argparse
import json
import math
import os
import random
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
    def __init__(self, grid_size: Tuple[int, int] = (128, 128), device: Optional[torch.device] = None):
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
                    steps_per_week: int = 3) -> Tuple[List[float], invRietkerk, dict, List[dict]]:

    if seed is not None:
        set_seed(seed)

    # Random start: each parameter log-uniform in rietkerk_model.INIT_RANGE (the same
    # range for every parameter, independent of SYNTHETIC_TRUTH),
    # redrawn until mean biomass stays within 0.1-10x of its initial level over the
    # training rollout (see rietkerk_model.draw_viable_model). Parameters are optimised in log space (see
    # invRietkerk), so learning_rate is a relative step size.
    sites = [(equilibrium_state, [weekly] * len(training_data[i]))
             for i, weekly in enumerate(weekly_precipitation_profiles)]
    model, init_draws = draw_viable_model(SYNTHETIC_TRUTH, sites, steps_per_week, device=device)
    model.init_draws = init_draws
    print(f"Random start accepted after {init_draws} draw(s)", flush=True)
    loss_function = nn.MSELoss()

    initial_params = model.parameter_values()
    parameter_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.95, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)
    loss_history = []

    def compute_loss():
        total_loss = 0
        for site_idx, weekly_precip in enumerate(weekly_precipitation_profiles):
            pred_surface_water = torch.zeros_like(equilibrium_state)
            pred_soil_water = torch.zeros_like(equilibrium_state)
            pred_biomass = equilibrium_state.clone()

            prev_pred_biomass = pred_biomass.clone()
            prev_target_biomass = pred_biomass.clone()

            for year_idx in range(len(training_data[site_idx])):
                pred_surface_water, pred_soil_water, pred_biomass = model.simulate_year_weekly(
                    pred_surface_water, pred_soil_water, pred_biomass,
                    weekly_precipitation=weekly_precip,
                    steps_per_week=steps_per_week
                )
                target_biomass = training_data[site_idx][year_idx]
                biomass_loss = loss_function((pred_biomass - prev_pred_biomass), (target_biomass - prev_target_biomass))
                prev_pred_biomass = pred_biomass
                prev_target_biomass = target_biomass
                total_loss += biomass_loss
        return total_loss

    def capture_parameters(epoch):
        return {'epoch': epoch, **model.parameter_values()}

    for epoch in range(num_epochs):
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

    if (num_epochs - 1) % save_interval != 0:
        parameter_history.append(capture_parameters(num_epochs - 1))

    return loss_history, model, initial_params, parameter_history


# ===========================================================================
#  3.  Main Batch Loop
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start run ID (inclusive)")
    parser.add_argument("--end", type=int, default=29, help="End run ID (inclusive)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "synthetic_invPDE_1site_rietkerk"), help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--grid_size", type=int, default=128)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/results", exist_ok=True)

    print("Generating training data...", flush=True)
    set_seed(42)  # Base seed so all runs share the same ground-truth

    # Same spin-up as the four-site experiment (400 mm/yr, inside the Turing range);
    # the single site gets 16.5/19 of it (347 mm/yr), the ratio of the original
    # script (spin-up 19, site 16.5).
    rain_multi = 400.0 / 19
    steps_per_week = 2
    data_generator = SyntheticDataGenerator(grid_size=(args.grid_size, args.grid_size), device=device)
    equilibrium_state = data_generator.generate_equilibrium_state(
        equilibrium_precipitation=19*rain_multi, equilibrium_years=100,
        steps_per_week=steps_per_week
    )

    n_sites = 1
    annual_totals = torch.linspace(16.5*rain_multi, 24.5*rain_multi, n_sites).tolist()
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

    summary = []

    for run_id in range(args.start, args.end + 1):
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
            steps_per_week=steps_per_week
        )

        elapsed = time.time() - start_time
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

        result_path = os.path.join(args.output_dir, "results", f"result_{run_id:02d}.json")
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


if __name__ == "__main__":
    main()
