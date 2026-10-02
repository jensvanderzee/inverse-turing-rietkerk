#!/usr/bin/env julia
"""
Extrapolation test: does a model fitted on ten years of one rainfall regime behave
sensibly 1000 years out, and under rainfall it never saw?

Port of `invPDE_1site_extrapolation.py`. The best one-site synthetic fit (lowest
finite final loss among `result_XX.json`) is rolled forward 1000 years from the
training equilibrium (spin-up seed 42, 400 mm/yr, 100 years, 2 steps/week) under
three seasonal regimes — below (280 mm), at (347 mm) and above (420 mm) the
training rainfall — next to the ground-truth model, which the Python script does
not draw.

Usage
-----
    julia --project=julia -t auto julia/scripts/extrapolation.jl

Options
    --runs <dir>      results of train_synthetic.jl --preset 1site
                      (default julia/results/synthetic_invPDE_1site_rietkerk/results)
    --years 1000
    --grid 128
    --out <dir>       default <runs>/../extrapolation_test
"""

using InverseTuring
using Printf
using Plots
using Random
import Statistics

include(joinpath(@__DIR__, "common.jl"))

const RUNS = string(argval("runs", joinpath(OUT_ROOT, "synthetic_invPDE_1site_rietkerk", "results")))
const YEARS = argint("years", 1000)
const GRID = argint("grid", 128)
const OUTDIR = string(argval("out", joinpath(dirname(RUNS), "extrapolation_test")))
const REGIMES = [(label = "below range", annual = 280.0), (label = "in range", annual = 347.0),
                 (label = "above range", annual = 420.0)]

banner("Extrapolation of the one-site fit")
report_threads()
files = filter(f -> occursin(r"^result_\d+\.json$", f), readdir(RUNS))
isempty(files) && error("no result_XX.json in $RUNS (run train_synthetic.jl --preset 1site first)")
runs = [load_run(joinpath(RUNS, f)) for f in files]
usable = filter(r -> r["final_loss"] isa Real && isfinite(r["final_loss"]), runs)
isempty(usable) && error("no run with a finite final loss")
best = usable[argmin([r["final_loss"] for r in usable])]
fitted = RietkerkParams(Dict{String,Any}(best["final_params"]))
@printf("%d usable runs; best: run %d (seed %d, final loss %.4f)\n", length(usable), best["run_id"],
        best["seed"], best["final_loss"])
show(stdout, MIME"text/plain"(), fitted)

cfg = SimConfig(steps_per_week = 2)
eq = equilibrium_biomass(SYNTHETIC_TRUTH, (GRID, GRID); precipitation = 400.0, years = 100,
                         cfg = cfg, rng = Xoshiro(42))
@printf("initial biomass mean %.4f\n", Statistics.mean(eq))

banner("Rolling out $YEARS years")
results = Vector{Any}(undef, length(REGIMES))
Threads.@threads for k in eachindex(REGIMES)
    weekly = sinusoidal_weekly_precip(REGIMES[k].annual)
    out = Dict{Symbol,Any}()
    for (name, p) in ((:fitted, fitted), (:truth, SYNTHETIC_TRUTH))
        s = simstate(eq)
        traj = [Statistics.mean(eq)]
        simulate_years!(s, p, weekly, cfg, YEARS; callback = (_, st) -> push!(traj, Statistics.mean(st.biomass)))
        out[name] = (traj = traj, field = copy(s.biomass))
    end
    results[k] = out
end
for (reg, r) in zip(REGIMES, results)
    @printf("%-12s %5.0f mm: final mean biomass fitted %.4f, truth %.4f\n", reg.label, reg.annual,
            r[:fitted].traj[end], r[:truth].traj[end])
end

mkpath(OUTDIR)
plt = plot(layout = (1, length(REGIMES)), size = (1500, 420), dpi = 150)
for (k, (reg, r)) in enumerate(zip(REGIMES, results))
    plot!(plt[k], 0:YEARS, r[:fitted].traj, lw = 2, label = "fitted (run $(best["run_id"]))")
    plot!(plt[k], 0:YEARS, r[:truth].traj, lw = 2, ls = :dash, color = :black, label = "truth")
    vline!(plt[k], [10], color = :gray, ls = :dot, label = "train horizon")
    plot!(plt[k], title = "$(reg.label): $(Int(reg.annual)) mm/yr", xlabel = "year", ylabel = "mean biomass")
end
savefig(plt, joinpath(OUTDIR, "invPDE_1site_extrapolation.png"))
vmax = maximum(maximum(r[n].field) for r in results for n in (:fitted, :truth))
fields = plot(layout = (2, length(REGIMES)), size = (1200, 800), dpi = 150)
for (k, (reg, r)) in enumerate(zip(REGIMES, results))
    for (row, n) in enumerate((:fitted, :truth))
        heatmap!(fields[(row - 1) * length(REGIMES) + k], r[n].field, clims = (0, vmax), c = :YlGn,
                 yflip = true, axis = false, aspect_ratio = 1,
                 title = @sprintf("%s, %s\nmean %.3f", n, reg.label, Statistics.mean(r[n].field)))
    end
end
savefig(fields, joinpath(OUTDIR, "invPDE_1site_extrapolation_fields.png"))
println("\nDone. Outputs in ", OUTDIR)
