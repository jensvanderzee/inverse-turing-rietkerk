#!/usr/bin/env julia
"""
Fit the Rietkerk coefficients to the real satellite data.

Port of `realdata_train_invPDE.py`: four training sites (b, i, c, e), NDVI × 1500,
delta loss averaged over transitions, 3 steps per week, log-space Adam
(lr 0.01, betas 0.9/0.95, decay 0.999 per epoch, gradient clipping at 10), 7500
epochs, ten models with seeds `77 + 102·index`, each started from a random draw
that keeps vegetation alive. Finished models are skipped and unfinished ones
resume from their checkpoint, so rerunning the same command after a crash
continues where it stopped.

Usage
-----
    julia --project=julia -t auto julia/scripts/train_realdata.jl
    julia --project=julia -t auto julia/scripts/train_realdata.jl --models 0 3   # models 0, 1, 2

Options (defaults match the Python entry point)
    --models START STOP   train models START..STOP-1 of --num-models
    --num-models 10
    --epochs 7500
    --steps-per-week 3
    --lr 0.01
    --multiplier 1500     NDVI -> biomass scaling
    --sites b,i,c,e
    --backend enzyme      enzyme | forwarddiff
    --parallel runs       runs: one thread per model; sites: models in turn, threads over sites
    --out <dir>           default julia/results/real_data_rietkerk/models

Outputs (the layout of the Python script, JSON instead of pickle/.pth)
    data_info.json
    parameters/model_XX_params.json     parameter history (snapshots every 10 epochs)
    results/model_XX.json               run record: losses, initial/final parameters, draws
    results/training_summary[_XX-YY].json
    checkpoints/model_XX.json           full training state, for resuming
    results/loss_curves.png
"""

using InverseTuring
using Printf
using Plots
import LinearAlgebra
import Statistics

include(joinpath(@__DIR__, "common.jl"))

const NUM_MODELS = argint("num-models", 10)
const MODEL_IDS = argrange("models", 0:(NUM_MODELS - 1))
const EPOCHS = argint("epochs", 7500)
const STEPS_PER_WEEK = argint("steps-per-week", 3)
const LR = argfloat("lr", 0.01)
const MULTIPLIER = argfloat("multiplier", NDVI_TO_BIOMASS_MULTIPLIER)
const SITES = arglist("sites", ["b", "i", "c", "e"])
const BACKEND = gradient_backend(string(argval("backend", "enzyme")))
const PARALLEL = Symbol(argval("parallel", "runs"))
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "real_data_rietkerk", "models")))

banner("Rietkerk inverse PDE: real satellite data")
report_threads()
println("sites            : ", join(SITES, ", "))
println("models           : ", first(MODEL_IDS), "..", last(MODEL_IDS), " of ", NUM_MODELS)
println("epochs           : ", EPOCHS)
@printf("steps per week   : %d  (dt = %.3f days)\n", STEPS_PER_WEEK, DAYS_PER_YEAR / (52 * STEPS_PER_WEEK))
println("learning rate    : ", LR, " (log space)")
println("gradient         : ", nameof(typeof(BACKEND)))
println("output           : ", OUTDIR)
for d in ("parameters", "results", "checkpoints")
    mkpath(joinpath(OUTDIR, d))
end

# ---------------------------------------------------------------------------
banner("Loading data")
sites = load_sites(DATA_DIR, SITES; multiplier = MULTIPLIER)
for s in sites
    ys = years(s)
    @printf("  %-12s %2d years %d-%d  %dx%d  precip %.0f-%.0f mm\n", s.name, length(s),
            minimum(ys), maximum(ys), size(s)..., minimum(o.precipitation for o in s),
            maximum(o.precipitation for o in s))
end
stats = biomass_stats(sites)
@printf("  biomass (per-image mean): %.2f - %.2f, mean %.2f\n",
        stats.min_biomass, stats.max_biomass, stats.mean_biomass)

cfg = SimConfig(steps_per_week = STEPS_PER_WEEK)
reference = realdata_reference(MULTIPLIER)
problem = InverseProblem(sites, cfg; average = true)
println("\n", problem)

save_json(joinpath(OUTDIR, "data_info.json"), Dict(
    "selected_sites" => SITES,
    "locations" => [s.name for s in sites],
    "global_stats" => Dict(string(k) => v for (k, v) in pairs(stats)),
    "ndvi_to_biomass_multiplier" => MULTIPLIER,
    "use_delta_loss" => true,
    "weekly_precip_config" => Dict("steps_per_week" => STEPS_PER_WEEK,
                                   "total_steps_per_year" => 52 * STEPS_PER_WEEK,
                                   "time_step_days" => DAYS_PER_YEAR / (52 * STEPS_PER_WEEK),
                                   "use_weekly_precip" => true),
    "time_points_per_location" => Dict(s.name => length(s) for s in sites),
    "reference_parameters" => paramdict(reference),
))

# ---------------------------------------------------------------------------
banner("Training")
param_path(i) = joinpath(OUTDIR, "parameters", @sprintf("model_%02d_params.json", i))
todo = [i for i in MODEL_IDS if !isfile(param_path(i))]
for i in setdiff(MODEL_IDS, todo)
    println("Model $(i + 1)/$NUM_MODELS already finished; skipping")
end

traincfg = TrainConfig(REALDATA_TRAIN_CONFIG; epochs = EPOCHS, learning_rate = LR,
                       verbose = PARALLEL !== :runs)
seed(i) = 77 + i * 102

function fit(i, prob, tc)
    r = train(prob; cfg = tc, reference = reference, seed = seed(i), backend = BACKEND,
              checkpoint_path = joinpath(OUTDIR, "checkpoints", @sprintf("model_%02d.json", i)))
    save_json(param_path(i), r.parameter_history)
    save_run(joinpath(OUTDIR, "results", @sprintf("model_%02d.json", i)), r;
             metadata = Dict("model_id" => i, "use_delta_loss" => true,
                             "weekly_config" => Dict("steps_per_week" => STEPS_PER_WEEK,
                                                     "total_steps_per_year" => 52 * STEPS_PER_WEEK),
                             "sites" => SITES, "gradient_backend" => string(nameof(typeof(BACKEND)))))
    @printf("Model %d: final loss %.6f after %d epochs (%.1f min, start after %d draw(s))%s\n",
            i + 1, r.final_loss, r.epochs_run, r.elapsed_seconds / 60, r.init_draws,
            r.converged ? "" : "  DIVERGED")
    flush(stdout)
    return r
end

t0 = time()
results = Dict{Int,TrainResult}()
if PARALLEL === :runs && Threads.nthreads() > 1 && length(todo) > 1
    inner = rethread(problem, false)
    loss_and_gradient(inner, reference; backend = BACKEND)      # compile once, serially
    lk = ReentrantLock()
    Threads.@threads for i in todo
        r = fit(i, inner, traincfg)
        lock(() -> (results[i] = r), lk)
    end
else
    outer = rethread(problem, PARALLEL !== :none)
    for i in todo
        println("\nTraining model $(i + 1)/$NUM_MODELS (seed $(seed(i)))...")
        results[i] = fit(i, outer, traincfg)
    end
end
@printf("\nTrained %d model(s) in %.1f min\n", length(results), (time() - t0) / 60)

# ---------------------------------------------------------------------------
banner("Summary")
ok = sort([i for (i, r) in results if r.converged && isfinite(r.final_loss)])
println("Successful models: $(length(ok))/$(length(todo))")
if !isempty(ok)
    losses = [results[i].final_loss for i in ok]
    @printf("Final loss: mean %.6f  std %.6f  min %.6f  max %.6f\n", Statistics.mean(losses),
            Statistics.std(losses; corrected = false), minimum(losses), maximum(losses))
    tag = MODEL_IDS == 0:(NUM_MODELS - 1) ? "" : @sprintf("_%02d-%02d", first(MODEL_IDS), last(MODEL_IDS))
    save_json(joinpath(OUTDIR, "results", "training_summary$tag.json"), Dict(
        "total_models" => NUM_MODELS, "model_ids" => collect(MODEL_IDS),
        "successful_models" => length(ok), "selected_sites" => SITES, "optimizer" => "Adam",
        "data_type" => "real_satellite_data_weekly_precip",
        "ndvi_to_biomass_multiplier" => MULTIPLIER, "use_delta_loss" => true,
        "loss_type" => "delta-based MSE",
        "weekly_precip_config" => Dict("steps_per_week" => STEPS_PER_WEEK,
                                       "total_steps_per_year" => 52 * STEPS_PER_WEEK,
                                       "time_step_days" => DAYS_PER_YEAR / (52 * STEPS_PER_WEEK)),
        "statistics" => Dict("mean_loss" => Statistics.mean(losses),
                             "std_loss" => Statistics.std(losses; corrected = false),
                             "min_loss" => minimum(losses), "max_loss" => maximum(losses))))
    best = ok[argmin(losses)]
    println("\nBest model: $best")
    show(stdout, MIME"text/plain"(), results[best].params)

    plt = plot(xlabel = "epoch", ylabel = "delta-MSE loss", yscale = :log10, legend = false,
               title = "Real-data fits", size = (900, 550), dpi = 150)
    for i in ok
        plot!(plt, max.(results[i].loss_history, eps()), lw = 1, alpha = 0.7)
    end
    savefig(plt, joinpath(OUTDIR, "results", "loss_curves$tag.png"))
end
println("\nDone. Outputs in ", OUTDIR)
