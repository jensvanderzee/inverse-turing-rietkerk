#!/usr/bin/env julia
"""
How much do independent fits agree with each other?

Port of `realdata_parameter_analysis.py`. Reads the parameter histories of the
real-data fits (Julia `.json` or Python `.pkl`), drops runs whose final values are
non-finite or on a clamp bound (`degenerate_parameters` against the real-data
reference), and reports per coefficient the spread across runs, the deviation from
the Rietkerk reference, and the trajectories over training. The Turing diagnostic
is evaluated at the mean training rainfall and the composite at the mean training
biomass, as in Python.

Usage
-----
    julia --project=julia julia/scripts/parameter_analysis.jl

Options
    --history <dir>   model_XX_params.{json,pkl}; default
                      julia/results/real_data_rietkerk/models/parameters
                      (results/real_data_rietkerk/models/parameters for the Python fits)
    --sites b,i,c,e   training sites (mean biomass and rainfall)
    --max-models 100
    --filter analysis `analysis`: degenerate values only (as this script in Python);
                      `tier1`: also drop runs that stopped early (as the bifurcation script)
    --multiplier 1500
    --out <dir>       default julia/results/real_data_rietkerk/parameter_history_analysis

Outputs: `four_site_agreement_table.csv`, `four_site_final_parameter_values.csv`
(the table test_realdata.jl and bifurcation.jl read), and figures.
"""

using InverseTuring
using Printf
using Plots
import CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "common.jl"))

const HISTORY_DIR = string(argval("history", joinpath(OUT_ROOT, "real_data_rietkerk", "models", "parameters")))
const SITES = arglist("sites", ["b", "i", "c", "e"])
const MAX_MODELS = argint("max-models", 100)
const FILTER = Symbol(argval("filter", "analysis"))
FILTER in (:analysis, :tier1) || error("--filter must be analysis or tier1, got $FILTER")
const MULTIPLIER = argfloat("multiplier", NDVI_TO_BIOMASS_MULTIPLIER)
const OUTDIR = string(argval("out", joinpath(dirname(dirname(HISTORY_DIR)), "parameter_history_analysis")))
const REFERENCE = realdata_reference(MULTIPLIER)

banner("Parameter agreement analysis")
println("histories        : ", HISTORY_DIR)
isdir(HISTORY_DIR) || error("history directory not found: $HISTORY_DIR")
histories = load_parameter_histories(HISTORY_DIR)
isempty(histories) && error("no model_XX_params.{pkl,json} found in $HISTORY_DIR")
@printf("loaded %d parameter histories (%d-%d snapshots each)\n", length(histories),
        minimum(length, values(histories)), maximum(length, values(histories)))

kept, dropped = FILTER === :tier1 ? tier1_filter(histories, REFERENCE) : drop_degenerate(histories, REFERENCE)
println("\nFilter (:$FILTER):")
for (id, reason) in dropped
    println("  dropped model $id: $reason")
end
@printf("  kept %d of %d\n", length(kept), length(histories))
length(kept) >= 2 || error("not enough valid models for an agreement analysis")
selected = sort(kept)[1:min(MAX_MODELS, length(kept))]

banner("Reference levels")
B = mean_training_biomass(DATA_DIR, SITES; multiplier = MULTIPLIER)
R = mean_training_precipitation(DATA_DIR, SITES)
@printf("mean training biomass B = %.6f; mean training rainfall %.4f mm/day (%.0f mm/yr)\n", B, R, R * 365)

banner("Cross-run agreement")
finals = Dict(id => RietkerkParams(histories[id][end]) for id in selected)
values_by_name = Dict{String,Vector{Float64}}(
    n => [getfield(finals[id], i) for id in selected] for (i, n) in enumerate(PARAM_NAMES))
values_by_name["turing_value"] = [turing_value(finals[id], R) for id in selected]
values_by_name["composite_value"] = [composite_value(finals[id], B) for id in selected]
tbl = agreement_table(values_by_name; ground_truth = REFERENCE)
mkpath(OUTDIR)
CSV.write(joinpath(OUTDIR, "four_site_agreement_table.csv"), tbl)
show(stdout, MIME"text/plain"(), tbl[:, [:parameter, :n_models, :mean, :cv, :ground_truth,
                                         :mape_vs_gt_pct, :agreement_score]]; allcols = true)
println("\n")
write_parameter_table(joinpath(OUTDIR, "four_site_final_parameter_values.csv"), selected,
                      [finals[id] for id in selected];
                      extra = Dict("turing_value" => values_by_name["turing_value"],
                                   "composite_value" => values_by_name["composite_value"]))

println("In Rietkerk's units (m²/day, 1/day, mm, g/m²):")
phys = [to_physical_units(finals[id]; multiplier = MULTIPLIER) for id in selected]
for (i, n) in enumerate(PARAM_NAMES)
    v = [getfield(q, i) for q in phys]
    @printf("  %-30s median %.4g   (Rietkerk 2002: %.4g)\n", n, Statistics.median(v), getfield(RIETKERK_2002, i))
end
n_turing = count(<(0), values_by_name["turing_value"])
@printf("\n%d of %d runs are Turing-unstable at the mean training rainfall\n", n_turing, length(selected))

banner("Plotting")
plt = plot(layout = (4, 3), size = (1300, 1200), dpi = 150)
for (i, name) in enumerate(PARAM_NAMES)
    for id in selected
        h = histories[id]
        plot!(plt[i], [s["epoch"] for s in h], [s[name] for s in h], lw = 1, alpha = 0.7, label = "")
    end
    hline!(plt[i], [getfield(REFERENCE, i)], lc = :black, ls = :dash, label = "")
    plot!(plt[i], title = PRETTY_NAMES[name], titlefontsize = 9, yscale = :log10,
          xlabel = i > 8 ? "epoch" : "", legend = false)
end
plot!(plt[12], framestyle = :none)
savefig(plt, joinpath(OUTDIR, "four_site_parameter_histories.png"))

rows = filter(r -> r.parameter != "turing_value", tbl)
plt2 = bar(1:DataFrames.nrow(rows), rows.cv, legend = false, dpi = 150, size = (950, 520),
           xticks = (1:DataFrames.nrow(rows), replace.(rows.parameter, "_" => " ")), xrotation = 40,
           ylabel = "coefficient of variation across runs", title = "Cross-run agreement")
savefig(plt2, joinpath(OUTDIR, "four_site_agreement_summary.png"))
println("\nDone. Outputs in ", OUTDIR)
