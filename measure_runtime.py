"""Estimate wall-clock runtime per single run for each experiment.

Runs a small number of training epochs from each experiment's pipeline,
measures mean seconds/epoch, and extrapolates to the full target epoch
count used in the paper. Also records data-setup overhead separately.

Does NOT complete any full training run. Produces:
    runtime_estimates.csv  - one row per experiment
    runtime_estimates.md   - formatted markdown table

Existing `summary_*.json` files (for the 3 synthetic experiments) already
contain `elapsed_seconds` per completed run; this script's numbers can be
cross-checked against those. The 4-site real-data training never logged
its wall time, so only this estimate is available for that row.

Usage
-----
    python measure_runtime.py                    # all experiments
    python measure_runtime.py --only real        # only real-data
    python measure_runtime.py --timed 20         # 20 timed epochs
"""
#%%
import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

#%%
# --- Target epoch counts for a full run (matches current script defaults) ---
# train_invPDE_synthetic_batch.py:387  -> num_epochs=10000
# train_rcnn_batch.py:361               -> default --num-epochs=1000
# realdata_train_invPDE.py:906          -> num_epochs=7500
TARGET_EPOCHS = {
    "invPDE_synthetic_1site": 7500,
    "invPDE_synthetic_4site": 7500,
    "rcnn_synthetic_4site":   1000,
    "invPDE_real_4site":      7500,
}


@dataclass
class Result:
    name: str
    target_epochs: int
    timed_epochs: int
    setup_seconds: float
    mean_epoch_seconds: float
    estimated_full_run_seconds: float
    device: str
    notes: str = ""


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_epochs(step_fn: Callable[[], None], warmup: int, timed: int,
                label: str) -> float:
    """Run `warmup` untimed epochs, then `timed` timed epochs.
    Returns mean seconds per timed epoch."""
    for _ in tqdm(range(warmup), desc=f"  [{label}] warmup"):
        step_fn()
    _sync()
    t0 = time.perf_counter()
    for _ in tqdm(range(timed), desc=f"  [{label}] timed "):
        step_fn()
    _sync()
    return (time.perf_counter() - t0) / timed


# ----------------------------------------------------------------------
# 1. invPDE synthetic (shared code for 1-site and 4-site)
# ----------------------------------------------------------------------
def _make_invpde_synthetic_step(n_sites: int, device: torch.device):
    # The generator is imported from the 1-site script: the 4-site script runs its
    # batch at module level, so importing it would start a training run. Both
    # scripts generate data the same way.
    from train_invPDE_synthetic_batch_1site import SyntheticDataGenerator, set_seed
    from rietkerk_model import invRietkerk, SYNTHETIC_TRUTH, generate_weekly_precipitation

    # 400 mm/yr spin-up in both synthetic scripts; the 1-site script's site is
    # 16.5/19 of it, the 4-site script's sites 15/21 ... 27/21.
    rain_multi = 400.0 / 21
    steps_per_week = 2
    set_seed(42)
    t0 = time.perf_counter()

    gen = SyntheticDataGenerator(grid_size=(128, 128), device=device)
    equilibrium_state = gen.generate_equilibrium_state(
        equilibrium_precipitation=21 * rain_multi, equilibrium_years=100,
        steps_per_week=steps_per_week,
    )

    if n_sites == 1:
        annual_totals = [16.5 * 400.0 / 19]
    else:
        annual_totals = torch.linspace(15 * rain_multi, 27 * rain_multi, n_sites).tolist()
    peak_weeks = [26.0] * n_sites
    weekly_profiles = [
        generate_weekly_precipitation(at, pw, amplitude_fraction=0.7)
        for at, pw in zip(annual_totals, peak_weeks)
    ]
    training_data = gen.generate_training_data(
        weekly_precipitation_profiles=weekly_profiles,
        equilibrium_state=equilibrium_state,
        time_series_years=10,
        noise_level=0.05,
        steps_per_week=steps_per_week,
    )

    set_seed(43)
    model = invRietkerk(trainable=True, reference=SYNTHETIC_TRUTH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.95), eps=1e-8)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9999)
    # Sites share one grid and are simulated as one batch, as in training.
    weekly_by_site = [list(week) for week in zip(*weekly_profiles)]
    targets = [torch.cat([training_data[s][y] for s in range(n_sites)]) for y in range(10)]
    initial = equilibrium_state.expand(n_sites, -1, -1, -1)
    _sync()
    setup_seconds = time.perf_counter() - t0

    def step():
        opt.zero_grad()
        psw = torch.zeros_like(initial)
        pss = torch.zeros_like(initial)
        pb = initial.clone()
        prev_pb = pb.clone()
        prev_tb = pb.clone()
        total_loss = 0
        for yr in range(10):
            psw, pss, pb = model.simulate_year_weekly(
                psw, pss, pb, weekly_by_site, steps_per_week=steps_per_week,
            )
            tb = targets[yr]
            total_loss = total_loss + ((pb - prev_pb) - (tb - prev_tb)).pow(2).mean(dim=(1, 2, 3)).sum()
            prev_pb = pb
            prev_tb = tb
        total_loss.backward()
        opt.step()
        sched.step()
        model.clamp_parameters_()

    return step, setup_seconds


# ----------------------------------------------------------------------
# 2. RCNN synthetic 4-site
# ----------------------------------------------------------------------
def _make_rcnn_step(device: torch.device):
    from train_rcnn_batch import (
        SyntheticDataGenerator,
        generate_weekly_precipitation,
        RCNNBaseline,
        set_seed,
        EQUILIBRIUM_PRECIPITATION,
        SITE_PRECIPITATION,
        STEPS_PER_WEEK,
    )

    set_seed(42)
    t0 = time.perf_counter()

    gen = SyntheticDataGenerator(grid_size=(128, 128), device=device)
    equilibrium_state = gen.generate_equilibrium_state(
        equilibrium_precipitation=EQUILIBRIUM_PRECIPITATION, equilibrium_years=100,
    )
    n_sites = 4
    annual_totals = SITE_PRECIPITATION
    weekly_profiles = [
        generate_weekly_precipitation(at, 26.0, amplitude_fraction=0.7)
        for at in annual_totals
    ]
    training_data = gen.generate_training_data(
        weekly_precipitation_profiles=weekly_profiles,
        equilibrium_state=equilibrium_state,
        time_series_years=10,
        noise_level=0.05,
        steps_per_week=STEPS_PER_WEEK,
    )

    torch.manual_seed(1000)
    model = RCNNBaseline(hidden_channels=32, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)
    loss_fn = nn.MSELoss()
    _sync()
    setup_seconds = time.perf_counter() - t0

    def step():
        opt.zero_grad()
        total = 0.0
        for site_idx, weekly in enumerate(weekly_profiles):
            pb = equilibrium_state.clone()
            hidden = None
            prev_pb = pb.clone()
            prev_tb = pb.clone()
            for yr in range(len(training_data[site_idx])):
                pb, hidden = model.simulate_year_weekly(pb, weekly, hidden)
                hidden = [h.detach() for h in hidden]
                tb = training_data[site_idx][yr]
                total = total + loss_fn(pb - prev_pb, tb - prev_tb)
                prev_pb = pb.detach()
                prev_tb = tb
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

    return step, setup_seconds


# ----------------------------------------------------------------------
# 3. invPDE real-data 4-site
# ----------------------------------------------------------------------
def _make_invpde_real_step(device: torch.device):
    from realdata_train_invPDE import RealDataLoader, invRietkerk
    from rietkerk_model import realdata_reference

    t0 = time.perf_counter()
    loader = RealDataLoader(
        data_dir="./data",
        selected_sites=["b", "i", "c", "e"],
        device=device,
        use_weekly_precip=True,
    )
    training_data = loader.get_training_data(ndvi_to_biomass_multiplier=1500.0)
    if len(training_data["location_time_series"]) == 0:
        raise RuntimeError("No real-data time series loaded — check ./data/ layout.")

    torch.manual_seed(77)
    model = invRietkerk(trainable=True, reference=realdata_reference(1500.0)).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.95), eps=1e-8)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.999)
    steps_per_week = 3  # matches __main__ config in realdata_train_invPDE.py
    _sync()
    setup_seconds = time.perf_counter() - t0

    def step():
        opt.zero_grad()
        total = 0
        num = 0
        for series in training_data["location_time_series"].values():
            if not series:
                continue
            initial_b = series[0]["biomass"].clone()
            psw = torch.zeros_like(initial_b, device=device)
            pss = torch.zeros_like(initial_b, device=device)
            pb = initial_b.clone()
            for t_idx in range(len(series) - 1):
                observed_delta = (
                    series[t_idx + 1]["biomass"] - series[t_idx]["biomass"]
                )
                initial_pb = pb.clone()
                psw, pss, pb = model.simulate_year_weekly(
                    psw, pss, pb,
                    weekly_precipitation=series[t_idx]["weekly_precipitation"],
                    steps_per_week=steps_per_week,
                )
                total = total + loss_fn(pb - initial_pb, observed_delta)
                num += 1
        (total / max(num, 1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()
        sched.step()
        model.clamp_parameters_()

    return step, setup_seconds


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
def measure(name: str, warmup: int, timed: int, device: torch.device) -> Result:
    print(f"\n=== {name} ===")

    if name == "invPDE_synthetic_1site":
        step, setup_s = _make_invpde_synthetic_step(n_sites=1, device=device)
    elif name == "invPDE_synthetic_4site":
        step, setup_s = _make_invpde_synthetic_step(n_sites=4, device=device)
    elif name == "rcnn_synthetic_4site":
        step, setup_s = _make_rcnn_step(device)
    elif name == "invPDE_real_4site":
        step, setup_s = _make_invpde_real_step(device)
    else:
        raise ValueError(name)

    print(f"  setup: {setup_s:.1f}s")
    mean_epoch_s = time_epochs(step, warmup, timed, label=name)
    target = TARGET_EPOCHS[name]
    est_total = mean_epoch_s * target
    print(f"  mean epoch: {mean_epoch_s:.3f}s  "
          f"-> {target} epochs ~= {est_total/60:.1f} min "
          f"({est_total/3600:.2f} h)")

    return Result(
        name=name,
        target_epochs=target,
        timed_epochs=timed,
        setup_seconds=setup_s,
        mean_epoch_seconds=mean_epoch_s,
        estimated_full_run_seconds=est_total,
        device=str(device),
    )


def _fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def write_outputs(results: list[Result], device_info: str):
    # CSV
    with open("runtime_estimates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment", "target_epochs", "timed_epochs",
            "setup_seconds", "mean_epoch_seconds",
            "estimated_full_run_seconds", "estimated_full_run_hms",
            "device", "notes",
        ])
        for r in results:
            w.writerow([
                r.name, r.target_epochs, r.timed_epochs,
                f"{r.setup_seconds:.2f}", f"{r.mean_epoch_seconds:.4f}",
                f"{r.estimated_full_run_seconds:.1f}",
                _fmt_hms(r.estimated_full_run_seconds),
                r.device, r.notes,
            ])

    # Markdown
    label = {
        "invPDE_synthetic_1site": "invPDE synthetic, 1 site",
        "invPDE_synthetic_4site": "invPDE synthetic, 4 sites",
        "rcnn_synthetic_4site":   "RCNN synthetic, 4 sites",
        "invPDE_real_4site":      "invPDE real data, 4 sites",
    }
    lines = []
    lines.append(f"# Estimated single-run training time\n")
    lines.append(f"Measured on: `{device_info}`  ")
    lines.append(f"Method: {results[0].timed_epochs} timed epochs per experiment, "
                 "extrapolated linearly to the target epoch count.\n")
    lines.append("| Experiment | Epochs/run | Mean s/epoch | Estimated wall time |")
    lines.append("|---|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {label.get(r.name, r.name)} | {r.target_epochs} "
            f"| {r.mean_epoch_seconds:.3f} | {_fmt_hms(r.estimated_full_run_seconds)} |"
        )
    with open("runtime_estimates.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\nWrote runtime_estimates.csv and runtime_estimates.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3,
                        help="Untimed warmup epochs (CUDA compilation, caches) (default 3)")
    parser.add_argument("--timed", type=int, default=10,
                        help="Timed epochs for extrapolation (default 10)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one: invPDE_synthetic_1site | "
                             "invPDE_synthetic_4site | rcnn_synthetic_4site | "
                             "invPDE_real_4site  (substring match allowed)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    # In Jupyter / IPython / VS Code interactive, sys.argv is the kernel's
    # launch args and will choke argparse. Fall back to defaults in that case.
    in_ipython = "ipykernel" in sys.modules or hasattr(sys, "ps1")
    args = parser.parse_args([] if in_ipython else None)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        device_info = f"CUDA: {torch.cuda.get_device_name(0)}  ({platform.platform()})"
    else:
        device_info = f"CPU: {platform.processor() or platform.machine()}  ({platform.platform()})"
    print(f"Device: {device_info}")
    print(f"Warmup: {args.warmup} epochs   Timed: {args.timed} epochs")

    all_names = list(TARGET_EPOCHS.keys())
    if args.only:
        names = [n for n in all_names if args.only.lower() in n.lower()]
        if not names:
            raise SystemExit(f"--only '{args.only}' matched nothing; options: {all_names}")
    else:
        names = all_names

    results = []
    for name in names:
        try:
            results.append(measure(name, args.warmup, args.timed, device))
        except Exception as e:
            print(f"  !! {name} failed: {e}")
            results.append(Result(
                name=name, target_epochs=TARGET_EPOCHS[name],
                timed_epochs=args.timed, setup_seconds=float("nan"),
                mean_epoch_seconds=float("nan"),
                estimated_full_run_seconds=float("nan"),
                device=str(device), notes=f"failed: {e}",
            ))

    write_outputs(results, device_info)


if __name__ == "__main__":
    main()

# %%
