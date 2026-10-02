#!/usr/bin/env julia
"""
Recover known parameters from synthetic data.

Port of `train_invPDE_synthetic_batch.py` (`--preset 4site`) and
`train_invPDE_synthetic_batch_1site.py` (`--preset 1site`). Data come from
`SYNTHETIC_TRUTH` (Rietkerk's values on 5 m cells, plant equation slowed 25×): a
100-year spin-up at 400 mm/yr from random fields, then ten years of seasonal weekly
rain per site, recorded once a year with noise 0.05. All eleven parameters are then
fitted from viable random starts with log-space Adam (lr 0.01, decay 0.9999, no
clipping; betas 0.9/0.95 for four sites, 0.95/0.99 for one), on the summed delta
loss at 2 steps per week. Finished runs are skipped; unfinished ones resume from
their checkpoint.

The data use Julia's RNG, so the noise realisation differs from the Python run;
the experiment is otherwise the same.

Usage
-----
    julia --project=julia -t auto julia/scripts/train_synthetic.jl --preset 4site --start 0 --end 9

Options
    --preset 4site|1site
    --start 0 --end 1    run ids (inclusive), seeds 42 + run id
    --epochs 10000
    --lr 0.01
    --grid 128
    --backend enzyme     enzyme | forwarddiff
    --checkpoint-every 50
    --out <dir>          default julia/results/synthetic_invPDE_<preset>_rietkerk

Outputs: `results/result_XX.json` (the Python `result_dict` keys),
`summary_XX-YY.json`, `checkpoints/run_XX.json`, `training_data_site_final.png`.
"""

using InverseTuring
using Printf
using Plots
using Random
import Statistics

include(joinpath(@__DIR__, "common.jl"))

const PRESET = string(argval("preset", "4site"))
PRESET in ("4site", "1site") || error("--preset must be 4site or 1site, got $PRESET")
const START = argint("start", 0)
const STOP = argint("end", 1)
const EPOCHS = argint("epochs", 10_000)
const LR = argfloat("lr", 0.01)
const GRID = argint("grid", 128)
const BACKEND = gradient_backend(string(argval("backend", "enzyme")))
const CHECKPOINT_EVERY = argint("checkpoint-every", 50)
const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "synthetic_invPDE_$(PRESET)_rietkerk")))

banner("Synthetic parameter recovery ($PRESET)")
report_threads()
println("grid             : $(GRID)x$(GRID)")
println("runs             : $START..$STOP")
println("epochs           : ", EPOCHS)
println("gradient         : ", nameof(typeof(BACKEND)))
println("output           : ", OUTDIR)
mkpath(joinpath(OUTDIR, "results"))
mkpath(joinpath(OUTDIR, "checkpoints"))

# ---------------------------------------------------------------------------
banner("Generating training data")
println("True parameters:")
show(stdout, MIME"text/plain"(), SYNTHETIC_TRUTH)
t0 = time()
ex = synthetic_experiment(; preset = PRESET == "4site" ? :four_site : :one_site,
                          grid = (GRID, GRID), rng = Xoshiro(42))
@printf("spin-up and rollouts: %.1f s\n", time() - t0)
eq = ex.equilibrium
@printf("equilibrium biomass: mean %.3f, std %.3f, range %.3f - %.3f\n",
        Statistics.mean(eq), Statistics.std(eq), minimum(eq), maximum(eq))
for (i, t) in enumerate(ex.annual_totals)
    @printf("  site %d: %.1f mm/yr  (Turing value at the mean rate: %.3g), final biomass mean %.3f\n",
            i, t, turing_value(SYNTHETIC_TRUTH, Statistics.mean(ex.profiles[i])),
            Statistics.mean(ex.targets[i][end]))
end
savefig(heatmap(ex.targets[end][end], aspect_ratio = 1, title = "site $(length(ex.targets)), final year"),
        joinpath(OUTDIR, "training_data_site_final.png"))
problem = ex.problem
println("\n", problem)

# ---------------------------------------------------------------------------
traincfg = PRESET == "4site" ? SYNTHETIC_TRAIN_CONFIG : SYNTHETIC_1SITE_TRAIN_CONFIG
traincfg = TrainConfig(traincfg; epochs = EPOCHS, learning_rate = LR)
summary = Dict{String,Any}[]
for run_id in START:STOP
    result_path = joinpath(OUTDIR, "results", @sprintf("result_%02d.json", run_id))
    if isfile(result_path)
        println("Run $run_id already finished ($result_path); skipping")
        continue
    end
    banner("Run $run_id")
    seed = 42 + run_id
    r = train(problem; cfg = traincfg, reference = SYNTHETIC_TRUTH, seed = seed, backend = BACKEND,
              checkpoint_path = joinpath(OUTDIR, "checkpoints", @sprintf("run_%02d.json", run_id)),
              checkpoint_every = CHECKPOINT_EVERY)
    save_run(result_path, r; metadata = Dict("run_id" => run_id, "ground_truth" => paramdict(SYNTHETIC_TRUTH),
                                             "gradient_backend" => string(nameof(typeof(BACKEND)))))
    println("  saved -> ", result_path)
    push!(summary, Dict("run_id" => run_id, "seed" => seed, "num_epochs" => r.epochs_run,
                        "final_loss" => r.final_loss, "elapsed_seconds" => r.elapsed_seconds))
    rel = (paramvector(r.params) .- paramvector(SYNTHETIC_TRUTH)) ./ paramvector(SYNTHETIC_TRUTH)
    for (n, e) in zip(PARAM_NAMES, rel)
        @printf("    %-30s relative error %+8.2f %%\n", n, 100e)
    end
end
save_json(joinpath(OUTDIR, @sprintf("summary_%02d-%02d.json", START, STOP)), summary)
losses = [s["final_loss"] for s in summary if isfinite(s["final_loss"])]
isempty(losses) || @printf("\n%d runs; final loss mean %.6f std %.6f\n", length(summary),
                           Statistics.mean(losses), Statistics.std(losses; corrected = false))
