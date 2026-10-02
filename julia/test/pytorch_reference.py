"""
Reference values from the PyTorch implementation (rietkerk_model.py) for the Julia
test suite.

Everything is computed in float64 on deterministic inputs (no RNG), so the Julia
tests can rebuild the same inputs and compare without PyTorch installed. Run from
the repository root:

    python julia/test/pytorch_reference.py

and commit the JSON it writes (julia/test/data/pytorch_reference.json). With
rasterio, pandas and scikit-learn installed it also scores the Rietkerk reference
parameters on a held-out site exactly as realdata_test_invPDE.py does.
"""
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import rietkerk_model as rm  # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(os.path.dirname(__file__), "data", "pytorch_reference.json")


def field(h, w, a, b, c, d):
    """Deterministic smooth positive field: a + b sin(0.7 i) cos(0.45 j) + c i + d j (0-based)."""
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    return a + b * np.sin(0.7 * i) * np.cos(0.45 * j) + c * i + d * j


def tensor(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64))[None, None]


def model_at(params, reference, semi_implicit=True, trainable=False):
    m = rm.invRietkerk(trainable=trainable, params=params, reference=reference,
                       semi_implicit=semi_implicit).double()
    # The constructor stores float32 logs; refill them at float64.
    return m.set_parameters(params)


def perturbed(p, factors):
    return {n: p[n] * f for n, f in zip(rm.PARAM_NAMES, factors)}


FACTORS = [1.3, 0.8, 1.2, 0.9, 1.4, 0.75, 1.1, 0.85, 1.25, 0.7, 1.15]


def main():
    ref = {}
    ref["param_names"] = rm.PARAM_NAMES
    ref["synthetic_truth"] = rm.SYNTHETIC_TRUTH
    ref["realdata_reference"] = rm.REALDATA_REFERENCE
    ref["realdata_reference_750"] = rm.realdata_reference(750.0)
    ref["physical_units_of_realdata_reference"] = rm.to_physical_units(rm.REALDATA_REFERENCE)
    lo_hi = rm.parameter_bounds(rm.REALDATA_REFERENCE)
    ref["bounds_realdata"] = {n: list(v) for n, v in lo_hi.items()}

    # degenerate_parameters on a few hand-made snapshots
    snaps = {
        "reference": dict(rm.REALDATA_REFERENCE),
        "on_lower_bound": {**rm.REALDATA_REFERENCE,
                           "mortality_rate": rm.REALDATA_REFERENCE["mortality_rate"] * 1.05e-4},
        "on_upper_bound": {**rm.REALDATA_REFERENCE,
                           "seepage_rate": rm.REALDATA_REFERENCE["seepage_rate"] * 0.95e4},
        "w0_at_one": {**rm.REALDATA_REFERENCE, "bare_soil_infiltration": 1.0},
        "nan": {**rm.REALDATA_REFERENCE, "plant_uptake_rate": float("nan")},
    }
    ref["degenerate"] = {k: rm.degenerate_parameters(v, rm.REALDATA_REFERENCE)
                         for k, v in snaps.items()}
    ref["degenerate_inputs"] = {k: {n: (None if math.isnan(x) else x) for n, x in v.items()}
                                for k, v in snaps.items()}

    # Forcing
    ref["weekly_350"] = rm.generate_weekly_precipitation(350.0)
    ref["weekly_420_flat"] = rm.generate_weekly_precipitation(420.0, amplitude_fraction=0.0)

    # Implicit diffusion on a 7x5 field
    u = field(7, 5, 3.0, 1.5, 0.2, -0.1)
    ref["diffusion_input"] = u.tolist()
    ref["diffusion_out_k0p7"] = rm.implicit_diffusion(tensor(u), torch.tensor(0.7))[0, 0].tolist()

    H, W = 12, 10
    B0 = field(H, W, 20.0, 10.0, 0.5, 0.3)
    O0 = field(H, W, 1.0, 0.5, 0.02, 0.01)
    W0 = field(H, W, 2.0, 0.8, -0.03, 0.05)
    ref["grid"] = [H, W]
    weekly = ref["weekly_350"]
    p_true = dict(rm.SYNTHETIC_TRUTH)
    p_pert = perturbed(p_true, FACTORS)
    ref["factors"] = FACTORS

    # One step of each scheme from (O0, W0, B0)
    for scheme, semi in (("semi_implicit", True), ("explicit", False)):
        m = model_at(p_pert, rm.SYNTHETIC_TRUTH, semi_implicit=semi)
        dt = 0.7 if not semi else 3.5
        o, w, b = m(tensor(O0), tensor(W0), tensor(B0), precipitation_rate=1.3, time_step=dt)
        ref[f"step_{scheme}"] = {"dt": dt, "rate": 1.3, "O": o[0, 0].tolist(),
                                 "W": w[0, 0].tolist(), "B": b[0, 0].tolist()}

    # One year, 2 steps per week, from water at zero
    m = model_at(p_pert, rm.SYNTHETIC_TRUTH)
    o, w, b = m.simulate_year_weekly(torch.zeros(1, 1, H, W), torch.zeros(1, 1, H, W),
                                     tensor(B0), weekly, steps_per_week=2)
    ref["year_spw2"] = {"O": o[0, 0].tolist(), "W": w[0, 0].tolist(), "B": b[0, 0].tolist()}

    # Delta loss over three transitions and its autograd gradient w.r.t. log params,
    # summed (synthetic scripts) and averaged (real-data script).
    targets = [field(H, W, 21.0 + k, 9.0 - k, 0.45 + 0.05 * k, 0.35 - 0.04 * k) for k in range(3)]
    forcings = [weekly, rm.generate_weekly_precipitation(280.0), rm.generate_weekly_precipitation(410.0)]
    ref["loss_targets"] = [t.tolist() for t in targets]
    ref["loss_forcings"] = forcings
    for delta in (True, False):
        for spw in (2, 3):
            m = model_at(p_pert, rm.SYNTHETIC_TRUTH, trainable=True)
            o = torch.zeros(1, 1, H, W)
            w = torch.zeros(1, 1, H, W)
            b = tensor(B0)
            prev_pred = b.clone()
            prev_target = b.clone()
            total = 0
            for k in range(3):
                o, w, b = m.simulate_year_weekly(o, w, b, forcings[k], steps_per_week=spw)
                tgt = tensor(targets[k])
                if delta:
                    total = total + torch.nn.functional.mse_loss(b - prev_pred, tgt - prev_target)
                else:
                    total = total + torch.nn.functional.mse_loss(b, tgt)
                prev_pred = b
                prev_target = tgt
            total.backward()
            key = f"loss_{'delta' if delta else 'absolute'}_spw{spw}"
            ref[key] = {"loss_sum": total.item(),
                        "grad_log_sum": {n: getattr(m, f"log_{n}").grad.item() for n in rm.PARAM_NAMES}}

    # Viability screen on the same inputs
    sites = [(tensor(B0), forcings)]
    ref["keeps_vegetation_truth"] = rm.keeps_vegetation(model_at(p_true, rm.SYNTHETIC_TRUTH), sites, 2)
    dead = {**p_true, "water_use_efficiency": p_true["water_use_efficiency"] * 0.01}
    ref["keeps_vegetation_dead"] = rm.keeps_vegetation(model_at(dead, rm.SYNTHETIC_TRUTH), sites, 2)

    # Linear stability
    precips = [0.9, 0.98, 1.0, 1.0015, 1.05, 1.15, 1.25, 1.258, 1.26, 1.3, 1.5]
    ref["turing_precips"] = precips
    ref["turing_truth"] = [rm.turing_value(p_true, r) for r in precips]
    ref["turing_wavelength_truth"] = [rm.turing_wavelength(p_true, r) for r in precips]
    ref["turing_pert"] = [rm.turing_value(p_pert, r) for r in precips]
    ref["steady_state_truth_1p1"] = rm.homogeneous_steady_state(p_true, 1.1)
    ref["jacobian_truth_1p1"] = rm.reaction_jacobian(p_true, ref["steady_state_truth_1p1"]).tolist()
    ref["composite_truth_B12"] = rm.composite_value(p_true, 12.0)
    ref["composite_realdata_B150"] = rm.composite_value(rm.REALDATA_REFERENCE, 150.0)

    # Held-out scoring on real data, as realdata_test_invPDE.py (float64)
    try:
        import realdata_train_invPDE as rt
        import realdata_test_invPDE as te
        te.DEVICE = torch.device("cpu")
        loader = rt.RealDataLoader(os.path.join(REPO, "data"), selected_sites=["f"],
                                   device=torch.device("cpu"), use_weekly_precip=True)
        data = loader.get_training_data(ndvi_to_biomass_multiplier=rm.NDVI_TO_BIOMASS_MULTIPLIER)
        ts = data["location_time_series"]["subsite_f"][:4]
        for entry in ts:
            entry["biomass"] = entry["biomass"].double()
        model = model_at(dict(rm.REALDATA_REFERENCE), rm.REALDATA_REFERENCE)
        ref["evaluate_subsite_f_4years_spw4"] = te.evaluate_model_on_site(model, ts)
        ref["evaluate_years"] = [e["year"] for e in ts]
    except ImportError as e:
        print(f"skipping the real-data check ({e})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(ref, f, indent=1, allow_nan=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
