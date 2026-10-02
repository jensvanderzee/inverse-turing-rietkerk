"""
    RietkerkParams{T}

The eleven coefficients of the Rietkerk et al. (2002) model that the inverse
problem fits, in natural units (not their logarithms):

| field                           | Rietkerk | meaning                                          |
|:--------------------------------|:---------|:-------------------------------------------------|
| `surface_water_diffusion_coeff` | D_O      | surface water flow (pixel²/day)                  |
| `soil_water_diffusion_coeff`    | D_W      | soil water diffusion (pixel²/day)                |
| `biomass_diffusion_coeff`       | D_P      | plant dispersal (pixel²/day)                     |
| `seepage_rate`                  | r_w      | soil water loss, evaporation + drainage (1/day)  |
| `mortality_rate`                | d        | plant mortality (1/day)                          |
| `infiltration_rate`             | α        | maximum infiltration rate (1/day)                |
| `plant_uptake_rate`             | g_max    | maximum specific water uptake                    |
| `water_use_efficiency`          | c        | water-to-biomass conversion                      |
| `infiltration_half_saturation`  | k₂       | biomass at which infiltration is half-way to α   |
| `bare_soil_infiltration`        | W₀       | infiltration on bare soil, as a fraction of α    |
| `uptake_half_saturation`        | k₁       | soil water at which uptake is half-maximal (mm)  |

Field order matches [`PARAM_NAMES`](@ref), which is the order of `PARAM_NAMES` in
`rietkerk_model.py` and therefore the column order of every parameter CSV and the
key order of every snapshot the Python scripts write — parameter vectors
round-trip between the two implementations without reordering.

`T` is free so that the struct can carry `Float64`, `Float32` or
`ForwardDiff.Dual` values; Enzyme differentiates it as an `Active` struct.
"""
struct RietkerkParams{T<:Real}
    surface_water_diffusion_coeff::T
    soil_water_diffusion_coeff::T
    biomass_diffusion_coeff::T
    seepage_rate::T
    mortality_rate::T
    infiltration_rate::T
    plant_uptake_rate::T
    water_use_efficiency::T
    infiltration_half_saturation::T
    bare_soil_infiltration::T
    uptake_half_saturation::T
end

"""
Names of the fitted parameters, in struct/vector order.

These strings are the wire format shared with the Python code (CSV headers, JSON
keys, parameter snapshots), so do not rename them.
"""
const PARAM_NAMES = (
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
)

const NPARAMS = length(PARAM_NAMES)

"""Plot labels in Rietkerk's notation, keyed like `PARAM_SYMBOLS` in Python."""
const PARAM_SYMBOLS = Dict(
    "surface_water_diffusion_coeff" => "D_O",
    "soil_water_diffusion_coeff" => "D_W",
    "biomass_diffusion_coeff" => "D_P",
    "seepage_rate" => "r_w",
    "mortality_rate" => "d",
    "infiltration_rate" => "α",
    "plant_uptake_rate" => "g_max",
    "water_use_efficiency" => "c",
    "infiltration_half_saturation" => "k₂",
    "bare_soil_infiltration" => "W₀",
    "uptake_half_saturation" => "k₁",
)

"""Readable names, as `PRETTY_NAMES` in Python."""
const PRETTY_NAMES = Dict(
    "surface_water_diffusion_coeff" => "Surface water diffusion",
    "soil_water_diffusion_coeff" => "Soil water diffusion",
    "biomass_diffusion_coeff" => "Biomass diffusion",
    "seepage_rate" => "Soil water loss rate",
    "mortality_rate" => "Mortality rate",
    "infiltration_rate" => "Max. infiltration rate",
    "plant_uptake_rate" => "Max. uptake rate",
    "water_use_efficiency" => "Water use efficiency",
    "infiltration_half_saturation" => "Infiltration half-sat. (k2)",
    "bare_soil_infiltration" => "Bare-soil infiltration (W0)",
    "uptake_half_saturation" => "Uptake half-sat. (k1)",
)

RietkerkParams(vals::NTuple{NPARAMS,T}) where {T<:Real} = RietkerkParams{T}(vals...)

function RietkerkParams(v::AbstractVector{T}) where {T<:Real}
    length(v) == NPARAMS ||
        throw(ArgumentError("expected $NPARAMS parameters, got $(length(v))"))
    RietkerkParams{T}(ntuple(i -> @inbounds(v[i]), NPARAMS)...)
end

"""
    RietkerkParams(; kwargs...)

Keyword constructor. Every parameter must be supplied; there is no silent default,
because a silently defaulted coefficient is indistinguishable from a fitted one.
"""
function RietkerkParams(; surface_water_diffusion_coeff, soil_water_diffusion_coeff,
                        biomass_diffusion_coeff, seepage_rate, mortality_rate,
                        infiltration_rate, plant_uptake_rate, water_use_efficiency,
                        infiltration_half_saturation, bare_soil_infiltration,
                        uptake_half_saturation)
    RietkerkParams(promote(surface_water_diffusion_coeff, soil_water_diffusion_coeff,
                           biomass_diffusion_coeff, seepage_rate, mortality_rate,
                           infiltration_rate, plant_uptake_rate, water_use_efficiency,
                           infiltration_half_saturation, bare_soil_infiltration,
                           uptake_half_saturation))
end

"""
    RietkerkParams(d::AbstractDict)

Build from a string-keyed dictionary: one row of a parameter CSV, a JSON object, or
a snapshot from a Python parameter history (extra keys such as `"epoch"` are
ignored).
"""
function RietkerkParams(d::AbstractDict)
    RietkerkParams(ntuple(NPARAMS) do i
        name = PARAM_NAMES[i]
        haskey(d, name) || throw(KeyError(name))
        Float64(d[name])
    end)
end

Base.eltype(::RietkerkParams{T}) where {T} = T
Base.eltype(::Type{RietkerkParams{T}}) where {T} = T

"""Fields as a tuple, in [`PARAM_NAMES`](@ref) order."""
astuple(p::RietkerkParams) = ntuple(i -> getfield(p, i), NPARAMS)

"""
    paramvector(p) -> Vector{T}

Flatten to the vector layout used by the optimiser and by the AD backends.
"""
paramvector(p::RietkerkParams{T}) where {T} = T[getfield(p, i) for i in 1:NPARAMS]

"""
    paramdict(p) -> Dict{String,Float64}

String-keyed dictionary in the format of every snapshot, CSV and JSON the Python
scripts write (`model.parameter_values()`).
"""
paramdict(p::RietkerkParams) =
    Dict{String,Float64}(PARAM_NAMES[i] => Float64(getfield(p, i)) for i in 1:NPARAMS)

Base.convert(::Type{RietkerkParams{T}}, p::RietkerkParams) where {T} =
    RietkerkParams(ntuple(i -> T(getfield(p, i)), NPARAMS))

"""
    map(f, p::RietkerkParams) -> RietkerkParams

Apply `f` to every coefficient, e.g. `map(log, p)`.
"""
Base.map(f, p::RietkerkParams) = RietkerkParams(map(f, astuple(p)))

"""
    map(f, p, q) -> RietkerkParams

Combine two parameter sets coefficient by coefficient, e.g. `map(/, fit, truth)`.
"""
Base.map(f, p::RietkerkParams, q::RietkerkParams) = RietkerkParams(map(f, astuple(p), astuple(q)))

function Base.show(io::IO, ::MIME"text/plain", p::RietkerkParams{T}) where {T}
    println(io, "RietkerkParams{", T, "}:")
    w = maximum(length, PARAM_NAMES)
    for i in 1:NPARAMS
        @printf(io, "  %-*s  %-5s  %.6g\n", w, PARAM_NAMES[i], PARAM_SYMBOLS[PARAM_NAMES[i]],
                Float64(ForwardDiff.value(getfield(p, i))))
    end
end

# ===========================================================================
#  Reference values and units
# ===========================================================================

"""
Rietkerk et al. (2002, p. 525) in their own units: m, day, mm, g/m². `d` is given
as a range (0–0.5); 0.25 is the value used for all their figures.
"""
const RIETKERK_2002 = RietkerkParams(
    surface_water_diffusion_coeff = 100.0,   # D_O, m²/d
    soil_water_diffusion_coeff = 0.1,        # D_W, m²/d
    biomass_diffusion_coeff = 0.1,           # D_P, m²/d
    seepage_rate = 0.2,                      # r_w, 1/d
    mortality_rate = 0.25,                   # d, 1/d
    infiltration_rate = 0.2,                 # α, 1/d
    plant_uptake_rate = 0.05,                # g_max, mm m² g⁻¹ d⁻¹
    water_use_efficiency = 10.0,             # c, g mm⁻¹ m⁻²
    infiltration_half_saturation = 5.0,      # k2, g/m²
    bare_soil_infiltration = 0.2,            # W0, –
    uptake_half_saturation = 5.0,            # k1, mm
)

"""Coefficients that carry the pixel size (pixel² per day)."""
const DIFFUSION_PARAMS = ("surface_water_diffusion_coeff", "soil_water_diffusion_coeff",
                          "biomass_diffusion_coeff")

"""
Coefficients that carry the biomass unit, with the power of the scale factor `s`:
Rietkerk's model is exactly invariant under B → sB, c → s·c, g_max → g_max/s,
k₂ → s·k₂, so a change of biomass unit only rescales these three.
"""
const BIOMASS_UNIT_POWER = ("plant_uptake_rate" => -1, "water_use_efficiency" => 1,
                            "infiltration_half_saturation" => 1)

"""
Factor by which `c`, `d` and `D_P` are divided in the reference sets (see
`PLANT_TIMESCALE_FACTOR` in `rietkerk_model.py`). With Rietkerk's d = 0.25/day plants
live ~4 days, and under weekly forcing with a dry season the published values wipe
out all vegetation in the first year. Dividing the plant equation by 25 leaves the
uniform equilibria, the Turing range and the pattern wavelength unchanged (a row
scaling preserves the sign of det(J − k²D)), so patterns persist through seasonal
forcing and the year-on-year changes the loss is fitted to stay informative.
"""
const PLANT_TIMESCALE_FACTOR = 25.0
const PLANT_TIMESCALE_PARAMS = ("water_use_efficiency", "mortality_rate", "biomass_diffusion_coeff")

"""One simulated year: 52 forcing weeks of 7 days."""
const DAYS_PER_YEAR = 364.0

"""Synthetic grids: 5 m cells resolve the ~45 m Rietkerk wavelength with ~9 cells."""
const SYNTHETIC_PIXEL_SIZE_M = 5.0
"""Real data: Landsat 30 m pixels."""
const REALDATA_PIXEL_SIZE_M = 30.0
"""Real-data biomass is NDVI × this multiplier."""
const NDVI_TO_BIOMASS_MULTIPLIER = 1500.0
"""Biomass = NDVI × 100 puts the training sites on Rietkerk's g/m² scale."""
const NDVI_TO_RIETKERK_GRAMS = 100.0

"""
    rietkerk_reference(; pixel_size_m = 5, biomass_scale = 1, plant_timescale = 25)
        -> RietkerkParams{Float64}

Rietkerk (2002) values converted to model units: diffusion coefficients to
pixel²/day for cells of `pixel_size_m`, the biomass-carrying coefficients to a unit
`biomass_scale` times g/m², and `c`, `d`, `D_P` divided by `plant_timescale`
(1 gives the published values).

The operations are applied in the order `rietkerk_reference` applies them in
Python, so the result is identical to the last bit.
"""
function rietkerk_reference(; pixel_size_m::Real = SYNTHETIC_PIXEL_SIZE_M,
                            biomass_scale::Real = 1.0,
                            plant_timescale::Real = PLANT_TIMESCALE_FACTOR)
    ref = paramdict(RIETKERK_2002)
    for name in PLANT_TIMESCALE_PARAMS
        ref[name] /= plant_timescale
    end
    for name in DIFFUSION_PARAMS
        ref[name] /= Float64(pixel_size_m)^2
    end
    for (name, power) in BIOMASS_UNIT_POWER
        ref[name] *= Float64(biomass_scale)^power
    end
    return RietkerkParams(ref)
end

"""
Ground truth of the synthetic experiments: Rietkerk's values on 5 m cells with the
plant time scale slowed by [`PLANT_TIMESCALE_FACTOR`](@ref).
"""
const SYNTHETIC_TRUTH = rietkerk_reference(; pixel_size_m = SYNTHETIC_PIXEL_SIZE_M,
                                           biomass_scale = 1.0)

"""
    realdata_reference(multiplier = 1500) -> RietkerkParams{Float64}

Reference point for the real data (initialisation centre and comparison values):
the synthetic truth on 30 m cells, in NDVI × `multiplier` biomass units.
"""
realdata_reference(multiplier::Real = NDVI_TO_BIOMASS_MULTIPLIER) =
    rietkerk_reference(; pixel_size_m = REALDATA_PIXEL_SIZE_M,
                       biomass_scale = multiplier / NDVI_TO_RIETKERK_GRAMS)

"""Reference set for the real data at the default NDVI multiplier."""
const REALDATA_REFERENCE = realdata_reference()

"""
    to_physical_units(p; pixel_size_m = 30, multiplier = 1500) -> RietkerkParams

Express fitted values in Rietkerk's units (m²/day, 1/day, mm, g/m²), undoing the
pixel size and the biomass scale (NDVI × 100 ≈ g/m²). Because of the biomass-unit
symmetry, fits made with different NDVI multipliers describe the same dynamics
exactly when they agree in these units.
"""
function to_physical_units(p::RietkerkParams; pixel_size_m::Real = REALDATA_PIXEL_SIZE_M,
                           multiplier::Real = NDVI_TO_BIOMASS_MULTIPLIER)
    scale = multiplier / NDVI_TO_RIETKERK_GRAMS
    out = paramdict(p)
    for name in DIFFUSION_PARAMS
        out[name] *= Float64(pixel_size_m)^2
    end
    for (name, power) in BIOMASS_UNIT_POWER
        out[name] /= Float64(scale)^power
    end
    return RietkerkParams(out)
end

"""
    parameter_bounds(reference; decades = 4) -> (lo, hi)

Clamp range for each parameter: `decades` orders of magnitude either side of the
reference, and W₀ ≤ 1 because it is a fraction. A parameter sitting on its lower
bound at the end of a run is a collapsed fit, not a small value.
"""
function parameter_bounds(reference::RietkerkParams; decades::Real = 4.0)
    lo = map(v -> v * 10.0^(-decades), reference)
    hi = map(v -> v * 10.0^decades, reference)
    hi = RietkerkParams(ntuple(i -> PARAM_NAMES[i] == "bare_soil_infiltration" ? 1.0 :
                                    getfield(hi, i), NPARAMS))
    return lo, hi
end

"""
    log_bounds(reference; decades = 4) -> (lo, hi)

[`parameter_bounds`](@ref) as vectors of logarithms: the box the optimiser, which
works on `log θ`, is projected onto after every step.
"""
function log_bounds(reference::RietkerkParams; decades::Real = 4.0)
    lo, hi = parameter_bounds(reference; decades = decades)
    return log.(paramvector(lo)), log.(paramvector(hi))
end

"""
    clamp_to_bounds(p, reference; decades = 4) -> RietkerkParams

Project into [`parameter_bounds`](@ref). This is what `invRietkerk` does when it
stores a value, so a model built from any dictionary sees the same numbers.
"""
function clamp_to_bounds(p::RietkerkParams, reference::RietkerkParams; decades::Real = 4.0)
    lo, hi = parameter_bounds(reference; decades = decades)
    return RietkerkParams(ntuple(i -> clamp(getfield(p, i), getfield(lo, i), getfield(hi, i)),
                                 NPARAMS))
end

"""
    degenerate_parameters(values, reference; margin = 1.1, bound_decades = 4) -> Vector{String}

Names of parameters that are non-finite or have run onto a clamp bound (within a
factor `margin`). `values` is a [`RietkerkParams`](@ref) or a snapshot dictionary;
missing keys count as non-finite. The upper bound of W₀ (1, a fraction) is a
legitimate value and is not flagged.
"""
function degenerate_parameters(values::AbstractDict, reference::RietkerkParams;
                               margin::Real = 1.1, bound_decades::Real = 4.0)
    lo, hi = parameter_bounds(reference; decades = bound_decades)
    bad = String[]
    for (i, name) in enumerate(PARAM_NAMES)
        v = Float64(get(values, name, NaN))
        l, h = getfield(lo, i), getfield(hi, i)
        if !isfinite(v) || v <= l * margin || (v >= h / margin && name != "bare_soil_infiltration")
            push!(bad, name)
        end
    end
    return bad
end

degenerate_parameters(p::RietkerkParams, reference::RietkerkParams; kwargs...) =
    degenerate_parameters(paramdict(p), reference; kwargs...)

"""
    randparams(rng, reference; decades = 1, bound_decades = 4) -> RietkerkParams{Float64}

Random start: every parameter log-uniform within `decades` orders of magnitude of
`reference`, then projected into the bounds (which only ever bites on W₀ ≤ 1), as
`invRietkerk(trainable=True)` draws it.

PyTorch's RNG stream cannot be reproduced, so a seed gives the same start in every
Julia run but not the start the Python script draws for that seed.
"""
function randparams(rng::Random.AbstractRNG, reference::RietkerkParams;
                    decades::Real = 1.0, bound_decades::Real = 4.0)
    u = 2 .* rand(rng, NPARAMS) .- 1
    p = RietkerkParams(ntuple(i -> getfield(reference, i) * 10.0^(decades * u[i]), NPARAMS))
    return clamp_to_bounds(p, reference; decades = bound_decades)
end
