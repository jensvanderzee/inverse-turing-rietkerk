# -*- coding: utf-8 -*-
"""
Rietkerk et al. (2002) dryland vegetation model as a differentiable PDE backbone.

Every training, testing and analysis script imports the model from here, so the
equations, parameter names, reference values and diagnostics exist in one place.

Model (Rietkerk et al. 2002, Am. Nat. 160:524, eq. 1; flat ground, no advection):

    dO/dt = D_O ΔO + R − α O (B + k2 W0)/(B + k2)
    dW/dt = D_W ΔW + α O (B + k2 W0)/(B + k2) − g_max W B/(W + k1) − r_w W
    dB/dt = D_P ΔB + c g_max W B/(W + k1) − d B

with surface water O (mm), soil water W (mm) and plant biomass B. In the notation
of Siero (2020, Physica D 414:132695, eqs. 1, 13–14) this is the same three-field
framework as the unified model this repository used before, with the linear
infiltration I(B) = B and uptake U(W,B) = W B replaced by the saturating forms
above and surface-water evaporation l1 = 0. The coefficients that play the same
role keep their old names, so result files stay comparable:

    name                            Rietkerk  Siero   meaning
    surface_water_diffusion_coeff   D_O       d1      surface water flow
    soil_water_diffusion_coeff      D_W       d2      soil water diffusion
    biomass_diffusion_coeff         D_P       d3      plant dispersal
    seepage_rate                    r_w       l2      soil water loss (evaporation + drainage)
    mortality_rate                  d         l3      plant mortality
    infiltration_rate               α         r2      maximum infiltration rate
    plant_uptake_rate               g_max     r1      maximum specific water uptake
    water_use_efficiency            c         j       water-to-biomass conversion
    infiltration_half_saturation    k2        —       biomass at which infiltration is half-way to α
    bare_soil_infiltration          W0        w0      infiltration on bare soil, as a fraction of α
    uptake_half_saturation          k1        k1      soil water at which uptake is half-maximal

Units. Time is in **days** (one simulated year is 52 weeks = 364 days), so a
weekly precipitation value in mm/day is injected as-is and fitted rates compare
directly with Rietkerk's per-day values. Space is in **pixels**: diffusion
coefficients are in pixel²/day and depend on the pixel size, see
`rietkerk_reference`. Biomass is in whatever unit the data uses; Rietkerk's model
has the exact symmetry B → sB, c → s·c, g_max → g_max/s, k2 → s·k2, so a change
of biomass unit only rescales those three coefficients.

Numerics. One step is a semi-implicit Euler sweep: sources (rain, infiltration
into the soil, growth) are explicit, while each field's own loss (infiltration out
of the surface, uptake + soil water loss, mortality) and its diffusion are
implicit. Surface water diffuses ~1000× faster than plants (D_O = 100 vs
D_P = 0.1 m²/day), and resolving the ~45 m pattern wavelength with an explicit
Laplacian would need a step of ~0.2 day; implicit diffusion removes that limit.
Implicit losses keep every field positive and the step stable for any parameter
values, so a random initialisation or an optimiser step can never turn a rollout
into NaN. The Laplacian is the same 5-point stencil with replicate boundaries as
before, solved exactly in its DCT-II eigenbasis. `semi_implicit=False` gives the
old fully explicit forward Euler step.

Like the old model, the fields are updated in sequence (O, then W using the new O,
then B using the new W).
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ===========================================================================
#  Parameters
# ===========================================================================
PARAM_NAMES = [
    "surface_water_diffusion_coeff",
    "soil_water_diffusion_coeff",
    "biomass_diffusion_coeff",
    "seepage_rate",
    "mortality_rate",
    "infiltration_rate",
    "plant_uptake_rate",
    "water_use_efficiency",
    "infiltration_half_saturation",
    "bare_soil_infiltration",
    "uptake_half_saturation",
]

# Plot labels, in Rietkerk's notation.
PARAM_SYMBOLS = {
    "surface_water_diffusion_coeff": r"$D_O$",
    "soil_water_diffusion_coeff": r"$D_W$",
    "biomass_diffusion_coeff": r"$D_P$",
    "seepage_rate": r"$r_w$",
    "mortality_rate": r"$d$",
    "infiltration_rate": r"$\alpha$",
    "plant_uptake_rate": r"$g_{max}$",
    "water_use_efficiency": r"$c$",
    "infiltration_half_saturation": r"$k_2$",
    "bare_soil_infiltration": r"$W_0$",
    "uptake_half_saturation": r"$k_1$",
}

PRETTY_NAMES = {
    "surface_water_diffusion_coeff": "Surface water diffusion",
    "soil_water_diffusion_coeff": "Soil water diffusion",
    "biomass_diffusion_coeff": "Biomass diffusion",
    "seepage_rate": "Soil water loss rate",
    "mortality_rate": "Mortality rate",
    "infiltration_rate": "Max. infiltration rate",
    "plant_uptake_rate": "Max. uptake rate",
    "water_use_efficiency": "Water use efficiency",
    "infiltration_half_saturation": "Infiltration half-sat. (k2)",
    "bare_soil_infiltration": "Bare-soil infiltration (W0)",
    "uptake_half_saturation": "Uptake half-sat. (k1)",
}

# Rietkerk et al. (2002), p. 525, in their units: m, day, mm, g/m².
# d is given as a range (0–0.5); 0.25 is the value used for all their figures.
RIETKERK_2002 = {
    "surface_water_diffusion_coeff": 100.0,   # D_O, m²/d
    "soil_water_diffusion_coeff": 0.1,        # D_W, m²/d
    "biomass_diffusion_coeff": 0.1,           # D_P, m²/d
    "seepage_rate": 0.2,                      # r_w, 1/d
    "mortality_rate": 0.25,                   # d, 1/d
    "infiltration_rate": 0.2,                 # α, 1/d
    "plant_uptake_rate": 0.05,                # g_max, mm m² g⁻¹ d⁻¹
    "water_use_efficiency": 10.0,             # c, g mm⁻¹ m⁻²
    "infiltration_half_saturation": 5.0,      # k2, g/m²
    "bare_soil_infiltration": 0.2,            # W0, –
    "uptake_half_saturation": 5.0,            # k1, mm
}

DIFFUSION_PARAMS = PARAM_NAMES[:3]
# Coefficients that carry the biomass unit, with the power of the scale factor s.
BIOMASS_UNIT_POWER = {
    "plant_uptake_rate": -1,
    "water_use_efficiency": 1,
    "infiltration_half_saturation": 1,
}

# Rietkerk's plants live on a ~4-day time scale (d = 0.25/day): his R is the
# climatic mean rainfall, and under weekly forcing with a dry season the published
# values wipe out all vegetation within the first year — even at ±30% seasonality
# (at the Niger sites, 29–32 of 52 weeks are dry). Dividing c, d and D_P by
# PLANT_TIMESCALE_FACTOR slows only the biomass equation: the uniform equilibria
# (O*, W*, B*), the Turing range (T1 = 1.001, T2 = 1.259 mm/day) and the pattern
# wavelength are unchanged, because the Turing condition depends on the sign of
# det(J − k²D), which a row scaling preserves. At 25 plants live ~100 days,
# patterns persist through the synthetic seasonal forcing and the year-on-year
# changes the loss is fitted to stay informative. Set it to 1 for the published
# values.
PLANT_TIMESCALE_FACTOR = 25.0
PLANT_TIMESCALE_PARAMS = ("water_use_efficiency", "mortality_rate", "biomass_diffusion_coeff")

# One simulated year: 52 forcing weeks of 7 days.
DAYS_PER_YEAR = 364.0

# Synthetic experiments: 5 m cells resolve the ~45 m Rietkerk wavelength with ~9
# cells, and biomass is in Rietkerk's own g/m².
SYNTHETIC_PIXEL_SIZE_M = 5.0
# Real data: Landsat 30 m pixels, biomass = NDVI × 1500.
REALDATA_PIXEL_SIZE_M = 30.0
NDVI_TO_BIOMASS_MULTIPLIER = 1500.0
# Biomass = NDVI × 100 puts the training sites on Rietkerk's scale (mean NDVI
# 0.08–0.17 → 8–17 g/m², 99th percentile ~0.3 → 30 g/m²; Rietkerk's patterns have
# means of ~10 and peaks of ~40 g/m²). With the ×1500 multiplier one data unit is
# therefore ~1/15 g/m².
NDVI_TO_RIETKERK_GRAMS = 100.0


def rietkerk_reference(pixel_size_m: float = SYNTHETIC_PIXEL_SIZE_M,
                       biomass_scale: float = 1.0,
                       plant_timescale: float = PLANT_TIMESCALE_FACTOR) -> Dict[str, float]:
    """Rietkerk (2002) values converted to model units.

    pixel_size_m: grid spacing in metres; diffusion coefficients become pixel²/day.
    biomass_scale: model biomass units per g/m² (e.g. 15 when biomass = NDVI × 1500
        and NDVI × 100 ≈ g/m²).
    plant_timescale: factor by which c, d and D_P are divided (see
        PLANT_TIMESCALE_FACTOR); 1 gives the published values.
    """
    ref = dict(RIETKERK_2002)
    for name in PLANT_TIMESCALE_PARAMS:
        ref[name] /= plant_timescale
    for name in DIFFUSION_PARAMS:
        ref[name] /= pixel_size_m ** 2
    for name, power in BIOMASS_UNIT_POWER.items():
        ref[name] *= biomass_scale ** power
    return ref


# Ground truth for the synthetic experiments: Rietkerk's values on 5 m cells, with
# the plant time scale slowed by PLANT_TIMESCALE_FACTOR.
SYNTHETIC_TRUTH = rietkerk_reference(SYNTHETIC_PIXEL_SIZE_M, 1.0)


def realdata_reference(ndvi_to_biomass_multiplier: float = NDVI_TO_BIOMASS_MULTIPLIER
                       ) -> Dict[str, float]:
    """Reference point for the real data (initialisation centre and comparison
    values): the synthetic truth on 30 m cells, in NDVI × multiplier biomass units."""
    return rietkerk_reference(REALDATA_PIXEL_SIZE_M,
                              ndvi_to_biomass_multiplier / NDVI_TO_RIETKERK_GRAMS)


REALDATA_REFERENCE = realdata_reference()


def to_physical_units(params: Dict[str, float],
                      pixel_size_m: float = REALDATA_PIXEL_SIZE_M,
                      ndvi_to_biomass_multiplier: float = NDVI_TO_BIOMASS_MULTIPLIER
                      ) -> Dict[str, float]:
    """Express fitted values in Rietkerk's units (m²/day, 1/day, mm, g/m²), undoing
    the pixel size and the biomass scale (NDVI × NDVI_TO_RIETKERK_GRAMS ≈ g/m²).

    Because of the biomass-unit symmetry, fits made with different NDVI multipliers
    describe the same dynamics exactly when they agree in these units."""
    scale = ndvi_to_biomass_multiplier / NDVI_TO_RIETKERK_GRAMS
    out = {name: float(params[name]) for name in PARAM_NAMES}
    for name in DIFFUSION_PARAMS:
        out[name] *= pixel_size_m ** 2
    for name, power in BIOMASS_UNIT_POWER.items():
        out[name] /= scale ** power
    return out


# Random starts: every parameter log-uniform in this range, the same for all of them
# and independent of any reference values.
INIT_RANGE = (0.01, 100.0)

# Parameters are stored as logarithms, so they are positive by construction; these
# limits only keep exp(log value) a finite, nonzero float32. They are not a prior.
NUMERICAL_BOUNDS = (1e-30, 1e30)


def parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """Clamp range for each parameter: NUMERICAL_BOUNDS for all of them."""
    return {name: NUMERICAL_BOUNDS for name in PARAM_NAMES}


def degenerate_parameters(values: Dict[str, float], reference: Dict[str, float],
                          margin: float = 1.1, decades: float = 4.0) -> List[str]:
    """Names of parameters that are non-finite or have drifted more than `decades`
    orders of magnitude (less a factor `margin`) from `reference`, as a collapsed
    fit does. An analysis filter only: fits are not clamped to this range. It
    replaces the old `value < 0.0011` / `<= 1e-4` checks, which would flag perfectly
    ordinary values here (D_W is ~1e-4 pixel²/day on 30 m pixels)."""
    bad = []
    for name in PARAM_NAMES:
        v = values.get(name, float("nan"))
        lo = reference[name] * 10.0 ** (-decades)
        hi = reference[name] * 10.0 ** decades
        if not np.isfinite(v) or v <= lo * margin or v >= hi / margin:
            bad.append(name)
    return bad


# ===========================================================================
#  Spatial operators
# ===========================================================================
_LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0, 1.0, 0.0]])


def laplacian(u: torch.Tensor) -> torch.Tensor:
    """5-point Laplacian with replicate boundaries (the old `Conv2d` operator)."""
    k = _LAPLACIAN_KERNEL.to(dtype=u.dtype, device=u.device)[None, None]
    return F.conv2d(F.pad(u, (1, 1, 1, 1), mode="replicate"), k)


_DCT_CACHE: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _dct_basis(h: int, w: int, dtype, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthonormal DCT-II matrices for lengths h and w, and the eigenvalues of −Δ
    (5-point, replicate boundaries) on the h × w grid:
    (2 − 2cos(πk/h)) + (2 − 2cos(πl/w))."""
    key = (h, w, dtype, device)
    if key not in _DCT_CACHE:
        def basis(n):
            k = torch.arange(n, dtype=torch.float64)[:, None]
            i = torch.arange(n, dtype=torch.float64)[None, :]
            c = torch.cos(math.pi * k * (2 * i + 1) / (2 * n)) * math.sqrt(2.0 / n)
            c[0] /= math.sqrt(2.0)
            lam = 2.0 - 2.0 * torch.cos(math.pi * torch.arange(n, dtype=torch.float64) / n)
            return c, lam
        ch, lh = basis(h)
        cw, lw = basis(w)
        mu = lh[:, None] + lw[None, :]
        _DCT_CACHE[key] = tuple(t.to(dtype=dtype, device=device) for t in (ch, cw, mu))
    return _DCT_CACHE[key]


def implicit_diffusion(u: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """Solve (I − coef·Δ) x = u exactly, Δ the 5-point Laplacian with replicate
    boundaries.

    The DCT-II basis diagonalises that operator (its eigenvalues are
    −(2 − 2cos(πk/h)) − (2 − 2cos(πl/w))), so the solve is a transform, a division
    and the inverse transform — here four small matrix products, which keeps the
    autograd graph no larger than the explicit convolution's. Differentiable in
    both `u` and `coef`.
    """
    ch, cw, mu = _dct_basis(*u.shape[-2:], u.dtype, u.device)
    spectrum = ch @ u @ cw.T
    spectrum = spectrum / (1.0 + coef * mu)
    return ch.T @ spectrum @ cw


# ===========================================================================
#  Model
# ===========================================================================
class invRietkerk(nn.Module):
    """Differentiable Rietkerk model.

    Trainable parameters are stored as logarithms (`log_<name>`), because the eleven
    coefficients span about five orders of magnitude; Adam's steps are then relative
    changes. Read and write natural values through `parameter_values()` and
    `set_parameters()`, or as attributes (`model.mortality_rate`).

    trainable=False builds a fixed model at `params` (default: `reference`, which
    defaults to SYNTHETIC_TRUTH).
    trainable=True without `params` draws every parameter log-uniformly from
    `init_range` (default INIT_RANGE, the same for all parameters; `reference` plays
    no part), using torch's global RNG so that `set_seed` fixes the initialisation.
    Values are only kept within NUMERICAL_BOUNDS.
    """

    def __init__(self, trainable: bool = False,
                 params: Optional[Dict[str, float]] = None,
                 reference: Optional[Dict[str, float]] = None,
                 init_range: Tuple[float, float] = INIT_RANGE,
                 semi_implicit: bool = True,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.trainable = trainable
        self.semi_implicit = semi_implicit
        # Store only weekly states for backprop and recompute each week on the way
        # back: roughly one extra forward pass for ~spw× less activation memory.
        self.gradient_checkpointing = gradient_checkpointing
        reference = dict(reference if reference is not None else SYNTHETIC_TRUTH)
        self.reference = reference
        self.bounds = parameter_bounds()

        if params is None and trainable:
            log_lo, log_hi = math.log10(init_range[0]), math.log10(init_range[1])
            u = torch.rand(len(PARAM_NAMES))
            params = {name: 10.0 ** (log_lo + (log_hi - log_lo) * u[i].item())
                      for i, name in enumerate(PARAM_NAMES)}
        elif params is None:
            params = reference
        missing = [n for n in PARAM_NAMES if n not in params]
        if missing:
            raise KeyError(f"missing parameters: {missing}")

        for name in PARAM_NAMES:
            lo, hi = self.bounds[name]
            value = torch.tensor([math.log(min(max(float(params[name]), lo), hi))])
            if trainable:
                setattr(self, f"log_{name}", nn.Parameter(value))
            else:
                self.register_buffer(f"log_{name}", value)

    # -- parameter access -------------------------------------------------------
    def __getattr__(self, name):
        if name in PARAM_NAMES:
            return torch.exp(super().__getattr__(f"log_{name}"))
        return super().__getattr__(name)

    def parameter_values(self) -> Dict[str, float]:
        """Natural-unit values, keyed by PARAM_NAMES (the format of every parameter
        snapshot, CSV and JSON written by the scripts)."""
        return {name: float(getattr(self, name).item()) for name in PARAM_NAMES}

    @torch.no_grad()
    def set_parameters(self, values: Dict[str, float]):
        for name in PARAM_NAMES:
            if name in values:
                lo, hi = self.bounds[name]
                v = min(max(float(values[name]), lo), hi)
                super().__getattr__(f"log_{name}").fill_(math.log(v))
        return self

    @torch.no_grad()
    def clamp_parameters_(self):
        """Keep every parameter inside its bounds (the old post-step clamp_)."""
        for name in PARAM_NAMES:
            lo, hi = self.bounds[name]
            super().__getattr__(f"log_{name}").clamp_(math.log(lo), math.log(hi))

    # -- dynamics ---------------------------------------------------------------
    def _update(self, field, source, loss_rate, diffusion, dt):
        """Advance `d field/dt = D Δfield + source − loss_rate · field` by one step.

        Semi-implicit: the loss is treated implicitly pointwise, then diffusion
        implicitly in Fourier space. Every factor is positive, so fields stay
        non-negative and the step is stable for any parameter values — which is
        what a fit needs, since the optimiser and the random initialisation both
        visit rates far from the reference. A uniform equilibrium (source =
        loss_rate · field) is an exact fixed point, as it is for explicit Euler.
        Explicit: the old forward Euler step.
        """
        if self.semi_implicit:
            return implicit_diffusion((field + dt * source) / (1.0 + dt * loss_rate),
                                      dt * diffusion)
        return field + dt * (diffusion * laplacian(field) + source - loss_rate * field)

    def coefficients(self) -> Dict[str, torch.Tensor]:
        """All eleven parameters as tensors (natural units), for passing to
        `forward` so that the exponentials are not recomputed every step."""
        return {name: getattr(self, name) for name in PARAM_NAMES}

    def forward(self, surface_water: torch.Tensor, soil_water: torch.Tensor,
                biomass: torch.Tensor, precipitation_rate: float = 0.0,
                time_step: float = 1.0, coefficients: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of length `time_step` days; `precipitation_rate` in mm/day."""
        p = coefficients if coefficients is not None else self.coefficients()
        alpha = p["infiltration_rate"]
        k2 = p["infiltration_half_saturation"]
        w0 = p["bare_soil_infiltration"]
        gmax = p["plant_uptake_rate"]
        k1 = p["uptake_half_saturation"]

        # max(·, 0) only matters for the explicit scheme, which can overshoot below
        # zero; there W + k1 or B + k2 could otherwise reach zero and turn the
        # rollout into NaN. The semi-implicit scheme keeps every field positive.
        b = biomass.clamp(min=0.0)
        infiltration_capacity = alpha * (b + k2 * w0) / (b + k2)   # α I(B), per unit O

        surface_water = self._update(
            surface_water, precipitation_rate, infiltration_capacity,
            p["surface_water_diffusion_coeff"], time_step)

        w = soil_water.clamp(min=0.0)
        uptake_capacity = gmax * b / (w + k1)                       # g_max B/(W + k1), per unit W
        soil_water = self._update(
            soil_water, infiltration_capacity * surface_water,
            uptake_capacity + p["seepage_rate"],
            p["soil_water_diffusion_coeff"], time_step)

        w = soil_water.clamp(min=0.0)
        growth = p["water_use_efficiency"] * gmax * w * b / (w + k1)
        biomass = self._update(
            biomass, growth, p["mortality_rate"],
            p["biomass_diffusion_coeff"], time_step)

        return surface_water, soil_water, biomass

    def simulate_year_weekly(self, surface_water: torch.Tensor, soil_water: torch.Tensor,
                             biomass: torch.Tensor, weekly_precipitation: Sequence[float],
                             steps_per_week: int = 3,
                             year_length: float = DAYS_PER_YEAR
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate one year under weekly forcing (rates in mm/day).

        Each week delivers `weekly_precipitation[w] * 7` mm, split evenly over its
        sub-steps. With the default `year_length` of 364 days the injected rate is
        exactly the weekly value in mm/day.

        `weekly_precipitation[w]` may also be a sequence with one rate per batch
        element (fields of shape (N, 1, H, W)), which simulates N sites of equal
        size at once — identical results, one kernel launch instead of N.
        """
        current_surface_water = surface_water.clone()
        current_soil_water = soil_water.clone()
        current_biomass = biomass.clone()

        if torch.is_tensor(weekly_precipitation):
            weekly_precipitation = weekly_precipitation.detach().cpu().numpy()
        weekly = np.asarray(weekly_precipitation, dtype=np.float64)
        if weekly.ndim == 1:
            weekly_precipitation = weekly.tolist()
        else:
            weekly_precipitation = torch.as_tensor(
                weekly, dtype=biomass.dtype, device=biomass.device)[:, :, None, None, None]

        num_weeks = len(weekly_precipitation)
        dt = year_length / (num_weeks * steps_per_week)

        coefficients = self.coefficients()

        def simulate_week(o, w, b, rate):
            for _ in range(steps_per_week):
                o, w, b = self.forward(o, w, b, precipitation_rate=rate, time_step=dt,
                                       coefficients=coefficients)
            return o, w, b

        checkpoint = self.gradient_checkpointing and torch.is_grad_enabled()
        for week in range(num_weeks):
            volume_per_step = weekly_precipitation[week] * 7.0 / steps_per_week
            adjusted_precip_rate = volume_per_step / dt
            state = (current_surface_water, current_soil_water, current_biomass)
            if checkpoint:
                state = torch.utils.checkpoint.checkpoint(
                    simulate_week, *state, adjusted_precip_rate, use_reentrant=False)
            else:
                state = simulate_week(*state, adjusted_precip_rate)
            current_surface_water, current_soil_water, current_biomass = state

        return current_surface_water, current_soil_water, current_biomass


@torch.no_grad()
def keeps_vegetation(model: invRietkerk, sites, steps_per_week: int,
                     min_fraction: float = 0.1, max_fraction: float = 10.0) -> bool:
    """Whether `model` keeps vegetation at a plausible level over a training rollout.

    `sites` is a list of (initial_biomass, weekly_precipitation_per_year) pairs,
    simulated from O = W = 0 exactly as the training loss does; the initial field
    may hold several sites along dim 0 (with per-site weekly rain). True if at the
    end every site's mean biomass lies between `min_fraction` and `max_fraction`
    times its initial mean — neither collapsed to bare soil nor exploded (the data
    stay within about ±20% on the real sites and 0.3–1.3× on the synthetic ones).
    """
    for initial, weekly_per_year in sites:
        o = torch.zeros_like(initial)
        w = torch.zeros_like(initial)
        b = initial.clone()
        for weekly in weekly_per_year:
            o, w, b = model.simulate_year_weekly(o, w, b, weekly, steps_per_week=steps_per_week)
        start = initial.mean(dim=(1, 2, 3))
        final = b.mean(dim=(1, 2, 3))
        ok = torch.isfinite(final) & (final > min_fraction * start) & (final < max_fraction * start)
        if not bool(torch.all(ok)):
            return False
    return True


def draw_viable_model(reference: Dict[str, float], sites, steps_per_week: int,
                      device=None, model_class=None, max_draws: int = 1000,
                      min_fraction: float = 0.1, max_fraction: float = 10.0,
                      **kwargs) -> Tuple[invRietkerk, int]:
    """Random start (as `invRietkerk(trainable=True)`) that keeps vegetation alive,
    and within a factor 10 of its initial level, over the training rollout; returns
    (model, number_of_draws).

    Rietkerk's bare state B = 0 is absorbing, and bistable with the vegetated one.
    From a start where the plants die, every later year contributes almost nothing
    to the gradient (B decays exponentially), so the fit can only tune the die-off:
    on the synthetic data 14 of 20 draws from the old ±1-decade prior were like that,
    and the fits started from such draws that were tried did not recover.
    Redrawing until the start sustains vegetation —
    restricting the prior to parameter sets compatible with the one thing the data
    show for certain, that there is vegetation — costs one forward rollout per draw
    and stays deterministic for a given seed. The upper limit drops starts where
    biomass explodes instead: on the real data those begin four orders of magnitude
    above the no-change loss.

    `model_class` builds the returned model (default invRietkerk); candidates are
    screened with invRietkerk itself, so a subclass that instruments
    simulate_year_weekly only sees the training run.
    """
    model = None
    for draw in range(1, max_draws + 1):
        model = invRietkerk(trainable=True, reference=reference, **kwargs)
        if device is not None:
            model = model.to(device)
        if keeps_vegetation(model, sites, steps_per_week, min_fraction, max_fraction):
            break
    else:
        import warnings
        warnings.warn(f"no random start kept vegetation at a plausible level in "
                      f"{max_draws} draws; using the last one")
    if model_class is not None and model_class is not invRietkerk:
        chosen = model.state_dict()
        model = model_class(trainable=True, reference=reference, **kwargs)
        model.load_state_dict(chosen)
        if device is not None:
            model = model.to(device)
    return model, draw


def model_from_parameters(params: Dict[str, float], device=None,
                          reference: Optional[Dict[str, float]] = None,
                          trainable: bool = False, **kwargs) -> invRietkerk:
    """Build a model at fixed parameter values (e.g. one row of a parameter CSV).

    `reference` is only stored with the model (no value is clipped: the clamp range
    is NUMERICAL_BOUNDS for every parameter).
    """
    model = invRietkerk(trainable=trainable, params=params,
                        reference=reference if reference is not None else params, **kwargs)
    return model.to(device) if device is not None else model


# ===========================================================================
#  Forcing
# ===========================================================================
def generate_weekly_precipitation(annual_total: float, peak_week: float = 26.0,
                                  amplitude_fraction: float = 0.7,
                                  num_weeks: int = 52) -> List[float]:
    """Sinusoidal weekly rain rates (mm/day) that sum to `annual_total` mm."""
    mean_rate = annual_total / 365.0
    raw = [mean_rate * (1.0 + amplitude_fraction *
                        math.cos(2.0 * math.pi * (w - peak_week) / num_weeks))
           for w in range(num_weeks)]
    raw_total = sum(r * 7.0 for r in raw)
    scale = annual_total / raw_total if raw_total > 0 else 1.0
    return [r * scale for r in raw]


# ===========================================================================
#  Linear stability diagnostics
# ===========================================================================
def homogeneous_steady_state(p: Dict[str, float], precip: float
                             ) -> Optional[Tuple[float, float, float]]:
    """Vegetated uniform equilibrium (O*, W*, B*) at constant rain `precip` (mm/day),
    or None if it does not exist (c·g_max ≤ d, or rain below r_w·W*)."""
    c, gmax, d = p["water_use_efficiency"], p["plant_uptake_rate"], p["mortality_rate"]
    if c * gmax <= d:
        return None
    w = p["uptake_half_saturation"] * d / (c * gmax - d)
    b = c / d * (precip - p["seepage_rate"] * w)
    if b <= 0:
        return None
    k2, w0 = p["infiltration_half_saturation"], p["bare_soil_infiltration"]
    o = precip / (p["infiltration_rate"] * (b + k2 * w0) / (b + k2))
    return o, w, b


def reaction_jacobian(p: Dict[str, float], state: Tuple[float, float, float]) -> np.ndarray:
    o, w, b = state
    alpha, k2, w0 = p["infiltration_rate"], p["infiltration_half_saturation"], p["bare_soil_infiltration"]
    gmax, k1, c = p["plant_uptake_rate"], p["uptake_half_saturation"], p["water_use_efficiency"]
    infil = (b + k2 * w0) / (b + k2)
    dinfil = k2 * (1.0 - w0) / (b + k2) ** 2
    du_dw = k1 * b / (w + k1) ** 2
    du_db = w / (w + k1)
    return np.array([
        [-alpha * infil, 0.0, -alpha * o * dinfil],
        [alpha * infil, -gmax * du_dw - p["seepage_rate"], alpha * o * dinfil - gmax * du_db],
        [0.0, c * gmax * du_dw, c * gmax * du_db - p["mortality_rate"]],
    ])


_MU = np.concatenate([[0.0], np.logspace(-6, math.log10(8.0), 600)])


def growth_rates(p: Dict[str, float], precip: float, mu: np.ndarray = _MU) -> Optional[np.ndarray]:
    """Largest real part of the linearised growth rate for each Laplacian eigenvalue
    −mu (pixel⁻²); mu ∈ [0, 8] covers every mode the 5-point grid can carry."""
    state = homogeneous_steady_state(p, precip)
    if state is None:
        return None
    J = reaction_jacobian(p, state)
    D = np.diag([p[n] for n in DIFFUSION_PARAMS])
    return np.linalg.eigvals(J[None] - mu[:, None, None] * D[None]).real.max(axis=1)


def turing_value(p: Dict[str, float], precip: float) -> float:
    """Sign diagnostic for Turing instability of the uniform vegetated state.

    Negative (same convention as before) means the uniform state is stable on its
    own but unstable to spatial perturbations, i.e. patterns can grow from it; its
    magnitude is the fastest spatial growth rate (1/day). Positive means no
    diffusion-driven instability. NaN when the uniform vegetated state does not
    exist or is already unstable without diffusion.

    Unlike the closed form used for the linear model, Rietkerk's uniform state
    depends on rainfall, so `precip` (mm/day) is required. Note that Rietkerk
    patterns also persist below the Turing range (between the limit point LP1 and
    T1, where the uniform state does not exist), so NaN does not mean "no patterns".
    """
    sigma = growth_rates(p, precip)
    if sigma is None or sigma[0] >= 0:
        return float("nan")
    return float(-sigma[1:].max())


def turing_wavelength(p: Dict[str, float], precip: float) -> float:
    """Wavelength (pixels) of the fastest-growing mode, or NaN if none grows."""
    sigma = growth_rates(p, precip)
    if sigma is None or sigma[0] >= 0 or sigma[1:].max() <= 0:
        return float("nan")
    return float(2.0 * math.pi / math.sqrt(_MU[1:][sigma[1:].argmax()]))


def composite_value(p: Dict[str, float], B: float) -> float:
    """Water-use efficiency of vegetation at biomass level B: the fraction of rain
    that infiltrates × the fraction of soil water that plants take up × c.

    Rietkerk has no surface-water loss (l1 = 0), so all rain eventually infiltrates
    and the first factor is 1. Uptake competes with soil water loss at the linear
    rates g_max·B/k1 vs r_w — the slope of uptake at W = 0, which is how Siero (2020,
    Table 1) maps Rietkerk onto the linear model — so this reduces to the old
    composite (r2B/(l1+r2B))·(r1B/(l2+r1B))·j with l1 = 0 and r1 = g_max/k1.
    """
    uptake = p["plant_uptake_rate"] * B / p["uptake_half_saturation"]
    return uptake / (p["seepage_rate"] + uptake) * p["water_use_efficiency"]
