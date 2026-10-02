#!/usr/bin/env julia
"""
Check the Julia port against the outputs of the PyTorch pipeline.

Two kinds of reference:

1. **Always available:** `julia/test/data/pytorch_reference.json`, written by
   `julia/test/pytorch_reference.py` from `rietkerk_model.py` itself (steps,
   rollouts, losses, autograd gradients, Turing diagnostics, a held-out score on
   real data). The test suite checks all of it; this script prints the headline
   numbers.
2. **If you have them:** the artefacts the Python scripts write under
   `results/real_data_rietkerk/` (not in the repository):
   - `models/data_info.json` — NDVI → biomass statistics
   - `parameter_history_analysis/four_site_final_parameter_values.csv` — the
     Turing and composite diagnostics from the same coefficients
   - `test_results/test_metrics.csv` — a 9-year rollout on the held-out sites,
     rescored here in `Float32`, PyTorch's precision
   - `parameter_history_analysis/four_site_agreement_table.csv` — recomputed from
     the pickled parameter histories

Nothing is re-fitted. Missing artefacts are skipped and reported.

Usage:
    julia --project=julia julia/scripts/validate_against_python.jl
"""

using InverseTuring
using Printf
import CSV, DataFrames, JSON, Statistics

include(joinpath(@__DIR__, "common.jl"))

const RD = joinpath(PY_RESULTS, "real_data_rietkerk")
const npass = Ref(0)
const nfail = Ref(0)
const skipped = String[]

function check(label, got, want; rtol)
    ok = isapprox(got, want; rtol = rtol) || (isnan(got) && isnan(want))
    ok ? (npass[] += 1) : (nfail[] += 1)
    rel = want == 0 ? abs(got) : abs(got - want) / abs(want)
    @printf("  %-50s %16.9g vs %16.9g  rel %.1e  %s\n", label, got, want, rel, ok ? "ok" : "FAIL")
    return ok
end

banner("1. Reference values from rietkerk_model.py")
ref = JSON.parsefile(joinpath(@__DIR__, "..", "test", "data", "pytorch_reference.json"); allownan = true)
testfield(h, w, a, b, c, d) = [a + b * sin(0.7i) * cos(0.45j) + c * i + d * j for i in 0:(h - 1), j in 0:(w - 1)]
B0 = testfield(12, 10, 20.0, 10.0, 0.5, 0.3)
tgts = [Float64.(permutedims(reduce(hcat, t))) for t in ref["loss_targets"]]
frc = [Float64.(f) for f in ref["loss_forcings"]]
p = RietkerkParams(paramvector(SYNTHETIC_TRUTH) .* Float64.(ref["factors"]))
for spw in (2, 3)
    prob = InverseProblem([SiteTrajectory(B0, copy(B0), frc, tgts)], SimConfig(steps_per_week = spw);
                          average = false, threaded = false)
    L, g = loss_and_gradient(prob, p)
    r = ref["loss_delta_spw$spw"]
    check("delta loss, $spw steps/week", L, r["loss_sum"]; rtol = 1e-10)
    glog = g .* paramvector(p)
    for (i, n) in enumerate(PARAM_NAMES)
        check("  ∂L/∂log $n", glog[i], r["grad_log_sum"][n]; rtol = 1e-7)
    end
end
for (R, want) in zip(ref["turing_precips"], ref["turing_truth"])
    want === nothing && continue
    check("turing_value(SYNTHETIC_TRUTH, $R)", turing_value(SYNTHETIC_TRUTH, R), want; rtol = 1e-8)
end

banner("2. Artefacts under results/real_data_rietkerk")
info_path = joinpath(RD, "models", "data_info.json")
if isfile(info_path)
    info = JSON.parsefile(info_path)
    sites = load_sites(DATA_DIR, ["b", "i", "c", "e"]; T = Float32)
    st = biomass_stats(sites)
    g = info["global_stats"]
    for k in ("min_biomass", "max_biomass", "mean_biomass", "std_biomass", "mean_precip")
        check("data_info $k", getfield(st, Symbol(k)), g[k]; rtol = 1e-6)
    end
else
    push!(skipped, info_path)
end

param_csv = joinpath(RD, "parameter_history_analysis", "four_site_final_parameter_values.csv")
if isfile(param_csv)
    ptab = read_parameter_table(param_csv)
    R = mean_training_precipitation(DATA_DIR, ["b", "i", "c", "e"])
    B = mean_training_biomass(DATA_DIR, ["b", "i", "c", "e"])
    for row in eachrow(ptab)
        q = params_from_row(row)
        check("turing_value, model $(row.model_id)", turing_value(q, R), row.turing_value; rtol = 1e-8)
        check("composite_value, model $(row.model_id)", composite_value(q, B), row.composite_value; rtol = 1e-6)
    end
    metrics_csv = joinpath(RD, "test_results", "test_metrics.csv")
    if isfile(metrics_csv)
        m = CSV.read(metrics_csv, DataFrames.DataFrame)
        cfg = SimConfig(steps_per_week = 4)
        test_sites = Dict(s.name => s for s in load_sites(DATA_DIR, ["f", "k", "j"]; T = Float32))
        for row in eachrow(m)
            haskey(test_sites, row.site) || continue
            q = convert(RietkerkParams{Float32},
                        params_from_row(only(filter(r -> r.model_id == row.model_id, eachrow(ptab)))))
            got = evaluate(q, test_sites[row.site], cfg)
            check("model $(row.model_id) $(row.site) delta-MSE", got.mse, row.mse; rtol = 1e-4)
            check("model $(row.model_id) $(row.site) correlation", got.correlation, row.correlation; rtol = 1e-4)
        end
    else
        push!(skipped, metrics_csv)
    end
else
    push!(skipped, param_csv)
end

hist_dir = joinpath(RD, "models", "parameters")
table_csv = joinpath(RD, "parameter_history_analysis", "four_site_agreement_table.csv")
if isdir(hist_dir) && isfile(table_csv)
    histories = load_parameter_histories(hist_dir)
    kept, _ = drop_degenerate(histories, REALDATA_REFERENCE)
    finals = [RietkerkParams(histories[id][end]) for id in kept]
    vals = Dict(n => [getfield(f, i) for f in finals] for (i, n) in enumerate(PARAM_NAMES))
    tbl = agreement_table(vals; ground_truth = REALDATA_REFERENCE)
    pyt = CSV.read(table_csv, DataFrames.DataFrame)
    for row in eachrow(tbl)
        r = findfirst(==(replace(row.parameter, "_" => " ") |> titlecase), pyt[!, 1])
        r === nothing && continue
        check("$(row.parameter) mean", round(row.mean; digits = 4), pyt[r, "Mean"]; rtol = 1e-4)
        check("$(row.parameter) CV", round(row.cv; digits = 4), pyt[r, "CV"]; rtol = 1e-3)
    end
else
    push!(skipped, table_csv)
end

banner("Summary")
@printf("%d checks passed, %d failed\n", npass[], nfail[])
isempty(skipped) || println("skipped (not found): \n  ", join(skipped, "\n  "))
nfail[] == 0 || exit(1)
