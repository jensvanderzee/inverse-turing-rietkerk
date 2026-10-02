"""
    homogeneous_steady_state(p, precip) -> (O, W, B) or nothing

Vegetated uniform equilibrium at constant rain `precip` (mm/day), or `nothing` if
it does not exist (c·g_max ≤ d, or rain below r_w·W*):

    W* = k₁ d/(c g_max − d),   B* = (c/d)(R − r_w W*),   O* = R / (α (B* + k₂W₀)/(B* + k₂))
"""
function homogeneous_steady_state(p::RietkerkParams, precip::Real)
    c, gmax, d = p.water_use_efficiency, p.plant_uptake_rate, p.mortality_rate
    c * gmax <= d && return nothing
    w = p.uptake_half_saturation * d / (c * gmax - d)
    b = c / d * (precip - p.seepage_rate * w)
    b <= 0 && return nothing
    k2, w0 = p.infiltration_half_saturation, p.bare_soil_infiltration
    o = precip / (p.infiltration_rate * (b + k2 * w0) / (b + k2))
    return (o, w, b)
end

"""
    reaction_jacobian(p, (O, W, B)) -> Matrix{Float64}

Jacobian of the reaction terms at a state.
"""
function reaction_jacobian(p::RietkerkParams, state)
    o, w, b = state
    α, k2, w0 = p.infiltration_rate, p.infiltration_half_saturation, p.bare_soil_infiltration
    gmax, k1, c = p.plant_uptake_rate, p.uptake_half_saturation, p.water_use_efficiency
    infil = (b + k2 * w0) / (b + k2)
    dinfil = k2 * (1 - w0) / (b + k2)^2
    du_dw = k1 * b / (w + k1)^2
    du_db = w / (w + k1)
    return [-α*infil 0.0 -α*o*dinfil
            α*infil (-gmax * du_dw - p.seepage_rate) (α * o * dinfil - gmax * du_db)
            0.0 c*gmax*du_dw (c * gmax * du_db - p.mortality_rate)]
end

"""
Laplacian eigenvalues `−μ` (pixel⁻²) at which the dispersion relation is sampled:
0 and 600 log-spaced values up to 8, which covers every mode the 5-point grid can
carry (`_MU` in `rietkerk_model.py`).
"""
const TURING_MU = vcat(0.0, 10.0 .^ range(-6, log10(8.0); length = 600))

"""
    growth_rates(p, precip; mu = TURING_MU) -> Vector{Float64} or nothing

Largest real part of the linearised growth rate of each mode `−μ` about the uniform
vegetated state, or `nothing` where that state does not exist.
"""
function growth_rates(p::RietkerkParams, precip::Real; mu::AbstractVector = TURING_MU)
    state = homogeneous_steady_state(p, precip)
    state === nothing && return nothing
    J = reaction_jacobian(p, state)
    D = LinearAlgebra.Diagonal([p.surface_water_diffusion_coeff, p.soil_water_diffusion_coeff,
                                p.biomass_diffusion_coeff])
    return [maximum(real, LinearAlgebra.eigvals(J - m * D)) for m in mu]
end

"""
    turing_value(p, precip) -> Float64

Sign diagnostic for Turing instability of the uniform vegetated state at rainfall
`precip` (mm/day). Negative means the uniform state is stable on its own but
unstable to spatial perturbations (patterns can grow), and its magnitude is the
fastest spatial growth rate (1/day); positive means no diffusion-driven
instability; `NaN` when the uniform state does not exist or is unstable without
diffusion. Port of `turing_value` — numerical, from the dispersion relation,
because Rietkerk's uniform state depends on rainfall.

Patterns also persist below the Turing range (between the limit point and T₁,
where the uniform state does not exist), so `NaN` does not mean "no patterns".
"""
function turing_value(p::RietkerkParams, precip::Real)
    σ = growth_rates(p, precip)
    (σ === nothing || σ[1] >= 0) && return NaN
    return -maximum(@view σ[2:end])
end

"""
    turing_wavelength(p, precip) -> Float64

Wavelength (pixels) of the fastest-growing mode, or `NaN` if none grows.
"""
function turing_wavelength(p::RietkerkParams, precip::Real)
    σ = growth_rates(p, precip)
    (σ === nothing || σ[1] >= 0 || maximum(@view σ[2:end]) <= 0) && return NaN
    return 2π / sqrt(TURING_MU[2:end][argmax(@view σ[2:end])])
end

"""
    composite_value(p, B) -> Float64

Water-use efficiency of vegetation at biomass `B`: the fraction of soil water that
plants take up, at the linear rates g_max·B/k₁ against r_w, times `c`. All rain
eventually infiltrates (no surface-water loss), so this is Siero's composite with
l₁ = 0 and r₁ = g_max/k₁. `B` is normally the mean training biomass.
"""
function composite_value(p::RietkerkParams, B::Real)
    uptake = p.plant_uptake_rate * B / p.uptake_half_saturation
    return uptake / (p.seepage_rate + uptake) * p.water_use_efficiency
end

"""
    mean_training_biomass(data_dir, sites; multiplier = 1500) -> Float64

Mean over all NDVI images of each image's mean biomass, for the given subsites —
the `B` of [`composite_value`](@ref). Computed in `Float32`, as in Python.
"""
function mean_training_biomass(data_dir::AbstractString, sites;
                               multiplier::Real = NDVI_TO_BIOMASS_MULTIPLIER)
    means = Float64[]
    for site in sites
        name = startswith(String(site), "subsite_") ? String(site) : "subsite_$site"
        dir = joinpath(data_dir, name, "$(name)_ndvi")
        isdir(dir) || (@warn "NDVI directory not found; skipping" dir; continue)
        for f in sort(filter(x -> endswith(lowercase(x), ".tif"), readdir(dir)))
            push!(means, Statistics.mean(ndvi_biomass(joinpath(dir, f);
                                                      multiplier = multiplier, T = Float32)))
        end
    end
    isempty(means) && throw(ArgumentError("no NDVI images found under $data_dir for $sites"))
    return Statistics.mean(means)
end

"""
    mean_training_precipitation(data_dir, sites) -> Float64

Mean weekly rainfall rate (mm/day) over the weekly ERA5 series of the given
subsites — the rainfall at which the real-data Turing diagnostic is evaluated.
"""
function mean_training_precipitation(data_dir::AbstractString, sites)
    rates = Float64[]
    for site in sites
        name = startswith(String(site), "subsite_") ? String(site) : "subsite_$site"
        path = joinpath(data_dir, name, "$(name)_precip", "$(name)_weekly_precip.csv")
        isfile(path) || continue
        append!(rates, Float64.(CSV.read(path, DataFrames.DataFrame).precipitation_mm_per_day))
    end
    isempty(rates) && throw(ArgumentError("no weekly precipitation found under $data_dir for $sites"))
    return Statistics.mean(rates)
end

"""
    tier1_filter(histories, reference; length_frac = 0.9) -> (kept, dropped)

Structural filter over `model_id => parameter_history`, as `filter_tier1` in
`bifurcation_parallel.py`: a run is dropped when its history is shorter than
`length_frac` of the longest (it stopped early), or when a final parameter is
non-finite or on its clamp bound ([`degenerate_parameters`](@ref)). Returns the
kept ids and `(id, reason)` pairs for the dropped ones.
"""
function tier1_filter(histories::AbstractDict, reference::RietkerkParams; length_frac::Real = 0.9)
    isempty(histories) && return (Int[], Tuple{Int,String}[])
    min_len = length_frac * maximum(length, values(histories))
    kept = Int[]
    dropped = Tuple{Int,String}[]
    for id in sort(collect(keys(histories)))
        history = histories[id]
        reasons = String[]
        length(history) < min_len &&
            push!(reasons, "short history ($(length(history)) < $(round(Int, min_len)) snapshots)")
        if isempty(history)
            push!(reasons, "empty history")
        else
            bad = degenerate_parameters(history[end], reference)
            isempty(bad) || push!(reasons, "degenerate final value(s): " *
                                           join(["$n=" * @sprintf("%.2e", get(history[end], n, NaN)) for n in bad], ", "))
        end
        isempty(reasons) ? push!(kept, id) : push!(dropped, (id, join(reasons, "; ")))
    end
    return (kept, dropped)
end

"""
    drop_degenerate(histories, reference) -> (kept, dropped)

The filter `realdata_parameter_analysis.py` applies: a run is dropped only if its
final snapshot is missing a parameter or has a degenerate one; history length is
ignored. Kept separate from [`tier1_filter`](@ref) because the two Python scripts
filter differently.
"""
function drop_degenerate(histories::AbstractDict, reference::RietkerkParams)
    kept = Int[]
    dropped = Tuple{Int,String}[]
    for id in sort(collect(keys(histories)))
        history = histories[id]
        if isempty(history)
            push!(dropped, (id, "empty history"))
        elseif !all(n -> haskey(history[end], n), PARAM_NAMES)
            push!(dropped, (id, "missing parameters in final snapshot"))
        else
            bad = degenerate_parameters(history[end], reference)
            isempty(bad) ? push!(kept, id) :
                push!(dropped, (id, "degenerate final value(s): " * join(bad, ", ")))
        end
    end
    return (kept, dropped)
end

"""
    agreement_table(values; ground_truth = nothing) -> DataFrame

Cross-run agreement per quantity (`name => fitted values`): mean, standard
deviation, coefficient of variation, range, and — where `ground_truth` has the
quantity — mean absolute percentage error and bias against it.
`agreement_score = 1/(1 + cv)`. Port of `calculate_parameter_agreement`.
"""
function agreement_table(values::AbstractDict{<:AbstractString,<:AbstractVector};
                         ground_truth::Union{Nothing,RietkerkParams} = nothing)
    gt = ground_truth === nothing ? Dict{String,Float64}() : paramdict(ground_truth)
    order = [n for n in PARAM_NAMES if haskey(values, n)]
    append!(order, sort([n for n in keys(values) if !(n in PARAM_NAMES)]))
    rows = NamedTuple[]
    for name in order
        v = Float64.(values[name])
        isempty(v) && continue
        mean_val = Statistics.mean(v)
        std_val = Statistics.std(v; corrected = false)
        cv = mean_val == 0 ? Inf : std_val / abs(mean_val)
        range_val = maximum(v) - minimum(v)
        gt_val = get(gt, name, NaN)
        push!(rows, (parameter = name, n_models = length(v), mean = mean_val, std = std_val,
                     cv = cv, min = minimum(v), max = maximum(v), range = range_val,
                     rel_range_pct = mean_val == 0 ? Inf : range_val / abs(mean_val) * 100,
                     ground_truth = gt_val,
                     mape_vs_gt_pct = isnan(gt_val) ? NaN : Statistics.mean(abs.((v .- gt_val) ./ gt_val)) * 100,
                     bias_vs_gt_pct = isnan(gt_val) ? NaN : Statistics.mean((v .- gt_val) ./ gt_val) * 100,
                     agreement_score = isfinite(cv) ? 1 / (1 + cv) : 0.0))
    end
    return DataFrames.DataFrame(rows)
end

"""
    bifurcation_sweep(p, initial_biomass, precip_values; years, cfg,
                      snapshot_at = Float64[], forcing = summer_weekly_precip)
        -> (means, snapshots)

Final-year mean biomass against annual rainfall for one parameter set: every level
is simulated independently for `years` years from the same initial field (with
`O = W = 0`), as `bifurcation_parallel.py` does. `snapshot_at` lists levels whose
final field is also returned. Levels run under `Threads.@threads`.

Every level starts from the same field, so this traces a single branch; to look for
hysteresis, sweep twice from a bare and a vegetated field.
"""
function bifurcation_sweep(p::RietkerkParams, initial_biomass::AbstractMatrix,
                           precip_values::AbstractVector; years::Integer, cfg::SimConfig,
                           snapshot_at::AbstractVector = Float64[],
                           forcing = summer_weekly_precip)
    means = zeros(Float64, length(precip_values))
    snapshots = Dict{Float64,Matrix{Float64}}()
    lk = ReentrantLock()
    Threads.@threads for i in eachindex(precip_values)
        precip = precip_values[i]
        state = simstate(initial_biomass)
        simulate_years!(state, p, forcing(precip), cfg, years)
        means[i] = Statistics.mean(state.biomass)
        if any(v -> isapprox(v, precip; atol = 1e-9), snapshot_at)
            field = Float64.(state.biomass)
            lock(lk) do
                snapshots[Float64(precip)] = field
            end
        end
    end
    return (means, snapshots)
end
