#!/usr/bin/env julia
"""
DifferentialEquations.jl, SciMLSensitivity.jl and Enzyme on the Rietkerk inverse
problem: what each buys, measured.

The Python pipeline integrates the PDE with a fixed-step semi-implicit scheme (2–3
steps per week) and differentiates it with autograd. The Julia package offers that
scheme (`SimConfig`, gradients by Enzyme or ForwardDiff) and the continuous PDE
(`ODEConfig`: any OrdinaryDiffEq solver, gradients by SciMLSensitivity's adjoints
with Enzyme VJPs, or ForwardDiff through the solver). This script compares them on
one transition (one year, one site) of two setups:

- **real**: site b (131×140, 30 m cells) at the real-data reference parameters;
- **synthetic**: a 64×64 spin-up on 5 m cells (`SYNTHETIC_TRUTH`), where surface-water
  diffusion makes the system stiff (8·D_O = 32/day).

Parameters are the references scaled by fixed factors, so the gradient is not
near zero. Reported:

1. `discretisation.csv` — error of the fixed-step scheme against the converged PDE
   in the loss and the gradient, by steps per week;
2. `solvers.csv` — forward cost and accuracy of fixed-step and adaptive solvers;
3. `gradients.csv` — cost and accuracy of every gradient method;
4. `summary.md` — the three tables.

Usage
-----
    julia --project=julia -t 1 julia/scripts/diffeq_comparison.jl [--implicit] [--only real|synthetic]

`--implicit` also times implicit solvers (KenCarp47, Rodas5P, FBDF) with a sparse
Jacobian; they are correct but 50–100× slower here, so they are off by default.
Expect ~30 min single-threaded (the first gradient of each kind compiles).
"""

using InverseTuring
using Printf
using Random
import CSV, DataFrames, Statistics, LinearAlgebra, Enzyme
using OrdinaryDiffEqTsit5, OrdinaryDiffEqStabilizedRK, OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqBDF
using SciMLSensitivity

include(joinpath(@__DIR__, "common.jl"))

const OUTDIR = string(argval("out", joinpath(OUT_ROOT, "diffeq_comparison")))
const ONLY = string(argval("only", "all"))
const IMPLICIT = argflag("implicit")
const FACTORS = [1.3, 0.8, 1.2, 0.9, 1.4, 0.75, 1.1, 0.85, 1.25, 0.7, 1.15]
LinearAlgebra.BLAS.set_num_threads(1)

banner("DifferentialEquations.jl / SciMLSensitivity.jl / Enzyme comparison")
mkpath(OUTDIR)

"""One-transition problem for each setup, with its parameters and tolerances."""
function setups()
    out = []
    if ONLY in ("all", "real")
        site = load_site(DATA_DIR, "b")
        o = site.observations
        tr = SiteTrajectory(copy(o[1].biomass), copy(o[1].biomass), [o[1].weekly_precipitation],
                            [copy(o[2].biomass)])
        push!(out, (name = "real (site b, 131x140, 30 m)", trs = [tr],
                    p = RietkerkParams(paramvector(REALDATA_REFERENCE) .* FACTORS),
                    tol = (abstol = 1e-3, reltol = 1e-4), tight = (abstol = 1e-7, reltol = 1e-10)))
    end
    if ONLY in ("all", "synthetic")
        cfg = SimConfig(steps_per_week = 2)
        eq = equilibrium_biomass(SYNTHETIC_TRUTH, (64, 64); precipitation = 400.0, years = 30,
                                 cfg = cfg, rng = Xoshiro(1))
        weekly = sinusoidal_weekly_precip(350.0)
        tgt = synthetic_series(SYNTHETIC_TRUTH, eq, weekly; years = 1, noise_level = 0.05, cfg = cfg,
                               rng = Xoshiro(2))
        tr = SiteTrajectory(copy(eq), copy(eq), [weekly], tgt)
        push!(out, (name = "synthetic (64x64, 5 m)", trs = [tr],
                    p = RietkerkParams(paramvector(SYNTHETIC_TRUTH) .* FACTORS),
                    tol = (abstol = 1e-5, reltol = 1e-4), tight = (abstol = 1e-9, reltol = 1e-10)))
    end
    return out
end

timed(f) = (f(); t = @elapsed r = f(); (r, t))      # second call: compilation excluded
relerr(a, b) = LinearAlgebra.norm(a .- b) / LinearAlgebra.norm(b)

disc_rows = NamedTuple[]
solver_rows = NamedTuple[]
grad_rows = NamedTuple[]
for S in setups()
    banner(S.name)
    p = S.p
    prob(cfg) = InverseProblem(S.trs, cfg; average = false, threaded = false)

    # Reference: the converged PDE, gradient by the interpolating adjoint at tight tolerance.
    ref = prob(ODEConfig(Tsit5(); S.tight...))
    (Lref, gref), tref = timed(() -> loss_and_gradient(ref, p; backend = AdjointODEBackend()))
    @printf("reference (Tsit5, reltol %.0e, InterpolatingAdjoint): loss %.10g  [%.1f s]\n",
            S.tight.reltol, Lref, tref)
    flush(stdout)

    println("\nFixed-step scheme (the Python discretisation), Enzyme gradient:")
    for spw in (1, 2, 3, 4, 8, 16, 48)
        pr = prob(SimConfig(steps_per_week = spw))
        L, tl = timed(() -> loss(p, pr))
        (Lg, g), tg = timed(() -> loss_and_gradient(pr, p; backend = EnzymeBackend()))
        @printf("  %2d steps/week (dt %.3f d): loss err %.2e, gradient err %.2e | loss %.3f s, gradient %.3f s\n",
                spw, 7 / spw, abs(L - Lref) / Lref, relerr(g, gref), tl, tg)
        flush(stdout)
        push!(disc_rows, (setup = S.name, steps_per_week = spw, dt_days = 7 / spw,
                          loss_rel_err = abs(L - Lref) / Lref, grad_rel_err = relerr(g, gref),
                          loss_seconds = tl, gradient_seconds = tg))
        push!(solver_rows, (setup = S.name, method = "fixed-step, $spw steps/week", tol = "-",
                            seconds = tl, loss_rel_err = abs(L - Lref) / Lref))
        spw in (2, 3) && push!(grad_rows, (setup = S.name, method = "fixed-step $spw/week + Enzyme reverse",
                                           seconds = tg, grad_rel_err = relerr(g, gref)))
    end
    pr3 = prob(SimConfig(steps_per_week = S.name[1:4] == "real" ? 3 : 2))
    (_, gf), tf = timed(() -> loss_and_gradient(pr3, p; backend = ForwardDiffBackend()))
    push!(grad_rows, (setup = S.name, method = "fixed-step $(pr3.cfg.steps_per_week)/week + ForwardDiff",
                      seconds = tf, grad_rel_err = relerr(gf, gref)))
    @printf("  ForwardDiff at %d steps/week: %.2f s\n", pr3.cfg.steps_per_week, tf)
    flush(stdout)

    println("\nAdaptive solvers (continuous PDE), forward:")
    algs = Any[("BS3", BS3(), false), ("Tsit5", Tsit5(), false), ("ROCK2", ROCK2(), false),
               ("ROCK4", ROCK4(), false)]
    IMPLICIT && append!(algs, [("KenCarp47 (sparse J)", KenCarp47(), true),
                               ("Rodas5P (sparse J)", Rodas5P(), true), ("FBDF (sparse J)", FBDF(), true)])
    for (name, alg, sparse) in algs
        pr = prob(ODEConfig(alg; sparse_jacobian = sparse, S.tol...))
        L, t = timed(() -> loss(p, pr))
        @printf("  %-22s abstol %.0e reltol %.0e: loss err %.2e | %.2f s\n", name, S.tol.abstol,
                S.tol.reltol, abs(L - Lref) / Lref, t)
        flush(stdout)
        push!(solver_rows, (setup = S.name, method = name, tol = "$(S.tol.abstol)/$(S.tol.reltol)",
                            seconds = t, loss_rel_err = abs(L - Lref) / Lref))
    end

    println("\nGradients of the continuous PDE (SciMLSensitivity with Enzyme VJPs, ForwardDiff):")
    ra = SciMLSensitivity.EnzymeVJP(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    best = S.name[1:4] == "real" ? ("BS3", BS3()) : ("ROCK4", ROCK4())
    for (aname, alg) in (best, ("Tsit5", Tsit5()))
        pr = prob(ODEConfig(alg; S.tol...))
        for (sname, sens) in (("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = EnzymeVJP())),
                              ("GaussAdjoint", GaussAdjoint(autojacvec = ra)),
                              ("QuadratureAdjoint", QuadratureAdjoint(autojacvec = ra)))
            (_, g), t = timed(() -> loss_and_gradient(pr, p; backend = AdjointODEBackend(sensealg = sens)))
            @printf("  %-6s + %-21s %7.2f s, gradient err %.2e\n", aname, sname, t, relerr(g, gref))
            flush(stdout)
            push!(grad_rows, (setup = S.name, method = "$aname + $sname (EnzymeVJP)", seconds = t,
                              grad_rel_err = relerr(g, gref)))
        end
    end
    pr = prob(ODEConfig(best[2]; S.tol...))
    (_, g), t = timed(() -> loss_and_gradient(pr, p; backend = ForwardDiffBackend()))
    @printf("  %-6s + ForwardDiff through the solver %.2f s, gradient err %.2e\n", best[1], t, relerr(g, gref))
    flush(stdout)
    push!(grad_rows, (setup = S.name, method = "$(best[1]) + ForwardDiff through the solver", seconds = t,
                      grad_rel_err = relerr(g, gref)))
end

disc = DataFrames.DataFrame(disc_rows)
solv = DataFrames.DataFrame(solver_rows)
grad = DataFrames.DataFrame(grad_rows)
CSV.write(joinpath(OUTDIR, "discretisation.csv"), disc)
CSV.write(joinpath(OUTDIR, "solvers.csv"), solv)
CSV.write(joinpath(OUTDIR, "gradients.csv"), grad)
open(joinpath(OUTDIR, "summary.md"), "w") do io
    println(io, "# Fixed-step scheme vs DifferentialEquations.jl\n")
    println(io, "One transition (one year, one site); errors are relative to the converged PDE.\n")
    for name in unique(disc.setup)
        println(io, "## ", name, "\n")
        println(io, "| method | time (s) | loss error | gradient error |")
        println(io, "|---|---:|---:|---:|")
        for r in eachrow(solv[solv.setup .== name, :])
            @printf(io, "| %s (forward) | %.3f | %.1e | |\n", r.method, r.seconds, r.loss_rel_err)
        end
        for r in eachrow(grad[grad.setup .== name, :])
            @printf(io, "| %s | %.3f | | %.1e |\n", r.method, r.seconds, r.grad_rel_err)
        end
        println(io)
    end
end
println("\nwrote ", OUTDIR)
