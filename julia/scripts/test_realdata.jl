#!/usr/bin/env julia
"""
Score fitted models on held-out sites.

Port of `realdata_test_invPDE.py`: every model in the parameter table is rolled
forward on sites it was not fitted to (f, k, j) at 4 steps per week, and scored
with the delta-MSE, delta-MAE and pixelwise correlation of
`evaluate_model_on_site`. The model with the lowest mean MSE is the one the
bifurcation analysis uses.

Usage
-----
    julia --project=julia -t auto julia/scripts/test_realdata.jl

Options
    --params <csv>        parameter table (default: the one parameter_analysis.jl writes,
                          julia/results/real_data_rietkerk/parameter_history_analysis/four_site_final_parameter_values.csv;
                          point it at results/... to score the Python fits)
    --sites f,k,j
    --steps-per-week 4
    --multiplier 1500
    --out <dir>           default julia/results/real_data_rietkerk/test_results
    --plot-best           also draw observed/predicted/difference maps for the best model

Writes `test_metrics.csv` and `test_summary.json` in the schema of the Python
script, so downstream scripts accept either.
"""

using InverseTuring
using Printf
using Plots
import CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "common.jl"))

const PARAMS = string(argval("params", joinpath(OUT_ROOT, "real_data_rietkerk", "parameter_history_analysis",
                                                "four_site_final_parameter_values.csv")))
const SITES = arglist("sites", ["f", "k", "j"])
const STEPS_PER_WEEK = argint("steps-per-week", 4)
const MULTIPLIER = argfloat("multiplier", NDVI_TO_BIOMASS_MULTIPLIER)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "real_data_rietkerk", "test_results")))

banner("Held-out evaluation")
report_threads()
isfile(PARAMS) || error("parameter table not found: $PARAMS (run parameter_analysis.jl first, or pass --params)")
ptab = read_parameter_table(PARAMS)
println("models           : ", DataFrames.nrow(ptab), " from ", PARAMS)
println("test sites       : ", join(SITES, ", "))
println("steps per week   : ", STEPS_PER_WEEK)
mkpath(OUTDIR)

sites = load_sites(DATA_DIR, SITES; multiplier = MULTIPLIER)
for s in sites
    ys = years(s)
    println("  ", s.name, ": ", length(s), " time points, years ", minimum(ys), "-", maximum(ys))
end
cfg = SimConfig(steps_per_week = STEPS_PER_WEEK)

rows = collect(eachrow(ptab))
jobs = [(i, j) for i in eachindex(rows) for j in eachindex(sites)]
metrics = Vector{Any}(undef, length(jobs))
t0 = time()
Threads.@threads for k in eachindex(jobs)
    i, j = jobs[k]
    metrics[k] = evaluate(params_from_row(rows[i]), sites[j], cfg)
end
@printf("evaluated %d models x %d sites in %.1f s\n", length(rows), length(sites), time() - t0)

has_turing = "turing_value" in DataFrames.names(ptab)
df = DataFrames.DataFrame(
    model_id = [rows[i].model_id for (i, _) in jobs],
    site = [sites[j].name for (_, j) in jobs],
    mse = [m.mse for m in metrics], mae = [m.mae for m in metrics],
    correlation = [m.correlation for m in metrics],
    num_transitions = [m.num_transitions for m in metrics],
    turing_value = [has_turing ? rows[i].turing_value : NaN for (i, _) in jobs])
CSV.write(joinpath(OUTDIR, "test_metrics.csv"), df)

banner("Test results")
summary = Dict{String,Any}()
for name in unique(df.site)
    sub = df[df.site .== name, :]
    st = Dict("mse_mean" => Statistics.mean(sub.mse), "mse_std" => Statistics.std(sub.mse),
              "mse_min" => minimum(sub.mse), "mse_max" => maximum(sub.mse),
              "mae_mean" => Statistics.mean(sub.mae), "mae_std" => Statistics.std(sub.mae),
              "corr_mean" => Statistics.mean(sub.correlation),
              "corr_std" => Statistics.std(sub.correlation), "num_models" => DataFrames.nrow(sub))
    summary[name] = st
    @printf("%s:\n  MSE:  %.2f +/- %.2f  (min=%.2f, max=%.2f)\n  MAE:  %.2f +/- %.2f\n  Corr: %.4f +/- %.4f\n",
            name, st["mse_mean"], st["mse_std"], st["mse_min"], st["mse_max"], st["mae_mean"],
            st["mae_std"], st["corr_mean"], st["corr_std"])
end
summary["overall"] = Dict("mse_mean" => Statistics.mean(df.mse), "mae_mean" => Statistics.mean(df.mae),
                          "corr_mean" => Statistics.mean(df.correlation))
save_json(joinpath(OUTDIR, "test_summary.json"), summary)

means = DataFrames.combine(DataFrames.groupby(df, :model_id), :mse => Statistics.mean => :mse)
best = means.model_id[argmin(means.mse)]
@printf("\nBest model by mean test MSE: model %d (MSE = %.2f)\n", best, minimum(means.mse))

p1 = plot(layout = (1, 3), size = (1500, 450), legend = false, dpi = 150)
for (k, (col, title)) in enumerate(((:mse, "MSE"), (:mae, "MAE"), (:correlation, "Correlation")))
    for (j, name) in enumerate(sort(unique(df.site)))
        vals = df[df.site .== name, col]
        scatter!(p1[k], fill(j, length(vals)) .+ 0.08 .* randn(length(vals)), vals, ms = 3, alpha = 0.6)
    end
    plot!(p1[k], xticks = (1:length(unique(df.site)), sort(unique(df.site))), title = title)
end
savefig(p1, joinpath(OUTDIR, "test_metrics.png"))

if argflag("plot-best")
    p = params_from_row(rows[findfirst(r -> r.model_id == best, rows)])
    for s in sites
        state = simstate(s.observations[1].biomass)
        ws = workspace(state)
        panels = Any[]
        for k in 1:(length(s) - 1)
            simulate_year!(state, p, s.observations[k].weekly_precipitation, cfg, ws)
            obs = s.observations[k + 1].biomass
            vmax = MULTIPLIER * 0.5
            push!(panels, heatmap(obs, clims = (0, vmax), title = "observed $(s.observations[k + 1].year)", yflip = true))
            push!(panels, heatmap(state.biomass, clims = (0, vmax), title = "predicted", yflip = true))
            d = state.biomass .- obs
            m = max(maximum(abs, d), 1e-6)
            push!(panels, heatmap(d, clims = (-m, m), c = :RdBu, title = "difference", yflip = true))
        end
        savefig(plot(panels..., layout = (length(s) - 1, 3), size = (1200, 330 * (length(s) - 1))),
                joinpath(OUTDIR, "best_model_predictions_$(s.name).png"))
    end
end
println("\nAll outputs saved to ", OUTDIR)
