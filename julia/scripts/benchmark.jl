#!/usr/bin/env julia
"""
Seconds per training epoch for each experiment and gradient backend, extrapolated
to a full run — the counterpart of `measure_runtime.py`.

One epoch is one loss-and-gradient evaluation plus an Adam step (the step is
negligible). Each configuration is warmed up once (compilation) and then timed.

Usage
-----
    julia --project=julia -t auto julia/scripts/benchmark.jl

Options
    --epochs 3            timed epochs per configuration
    --only real|synthetic restrict to one group
    --backends enzyme,forwarddiff
    --out <dir>           default julia/results/benchmark (runtime_estimates.csv / .md)
"""

using InverseTuring
using Printf
using Random
import CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "common.jl"))

const EPOCHS = argint("epochs", 3)
const ONLY = string(argval("only", "all"))
const BACKENDS = arglist("backends", ["enzyme", "forwarddiff"])
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "benchmark")))
const TARGET_EPOCHS = Dict("invPDE synthetic, 1 site" => 10_000, "invPDE synthetic, 4 sites" => 10_000,
                           "invPDE real data, 4 sites" => 7500)

banner("Training-time benchmark")
nthreads = report_threads()
println("CPU: ", Sys.cpu_info()[1].model, " (", Sys.CPU_THREADS, " threads)")

experiments = Pair{String,Any}[]
if ONLY in ("all", "synthetic")
    t0 = time()
    ex4 = synthetic_experiment(; preset = :four_site, rng = Xoshiro(42))
    ex1 = synthetic_experiment(; preset = :one_site, rng = Xoshiro(42))
    @printf("synthetic data generated in %.1f s\n", time() - t0)
    push!(experiments, "invPDE synthetic, 1 site" => (ex1.problem, SYNTHETIC_TRUTH))
    push!(experiments, "invPDE synthetic, 4 sites" => (ex4.problem, SYNTHETIC_TRUTH))
end
if ONLY in ("all", "real")
    sites = load_sites(DATA_DIR, ["b", "i", "c", "e"])
    push!(experiments, "invPDE real data, 4 sites" =>
                       (InverseProblem(sites, SimConfig(steps_per_week = 3)), REALDATA_REFERENCE))
end

rows = NamedTuple[]
for (name, (prob, ref)) in experiments, bname in BACKENDS
    backend = gradient_backend(bname)
    banner("$name — $bname")
    println(prob)
    p = ref
    cache = gradient_cache(prob, backend)
    g = zeros(NPARAMS)
    tc = @elapsed loss_and_gradient!(g, cache, p)
    @printf("  first call (compile + run): %.1f s\n", tc)
    times = Float64[]
    for _ in 1:EPOCHS
        push!(times, @elapsed loss_and_gradient!(g, cache, p))
    end
    tl = @elapsed loss(p, prob)
    per = Statistics.mean(times)
    full = per * TARGET_EPOCHS[name]
    @printf("  %.3f s/epoch (loss alone %.3f s, ratio %.1f) -> %d epochs ≈ %.1f h\n",
            per, tl, per / tl, TARGET_EPOCHS[name], full / 3600)
    push!(rows, (experiment = name, backend = bname, threads = nthreads, epochs_per_run = TARGET_EPOCHS[name],
                 timed_epochs = EPOCHS, mean_epoch_seconds = per, loss_seconds = tl,
                 gradient_over_loss = per / tl, estimated_hours = full / 3600, first_call_seconds = tc))
end

mkpath(OUTDIR)
df = DataFrames.DataFrame(rows)
CSV.write(joinpath(OUTDIR, "runtime_estimates.csv"), df)
open(joinpath(OUTDIR, "runtime_estimates.md"), "w") do io
    println(io, "# Estimated single-run training time (Julia)\n")
    println(io, "Measured on: `", Sys.cpu_info()[1].model, "`, ", nthreads, " Julia threads.  ")
    println(io, "Method: ", EPOCHS, " timed epochs after one warm-up, extrapolated linearly.\n")
    println(io, "| Experiment | Gradient | Epochs/run | s/epoch | gradient / loss | Estimated wall time |")
    println(io, "|---|---|---:|---:|---:|---:|")
    for r in rows
        @printf(io, "| %s | %s | %d | %.3f | %.1f | %.1f h |\n", r.experiment, r.backend, r.epochs_per_run,
                r.mean_epoch_seconds, r.gradient_over_loss, r.estimated_hours)
    end
end
println("\nwrote ", joinpath(OUTDIR, "runtime_estimates.csv"), " and .md")
