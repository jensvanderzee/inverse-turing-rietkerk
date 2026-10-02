using Test
using InverseTuring
using Random
import Statistics
import ForwardDiff
import Enzyme
import JSON
import SparseArrays
using OrdinaryDiffEqTsit5, OrdinaryDiffEqStabilizedRK
using SciMLSensitivity

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(REPO, "data")
const HAVE_DATA = isdir(joinpath(DATA_DIR, "subsite_b"))
const PYREF_PATH = joinpath(@__DIR__, "data", "pytorch_reference.json")
const PYREF = isfile(PYREF_PATH) ? JSON.parsefile(PYREF_PATH; allownan = true) : nothing

HAVE_DATA || @warn "data/ not found; data-dependent tests will be skipped" DATA_DIR
PYREF === nothing && @warn "no PyTorch reference values; run julia/test/pytorch_reference.py"

"""Deterministic smooth field, the `field` helper of pytorch_reference.py (0-based i, j)."""
testfield(h, w, a, b, c, d) = [a + b * sin(0.7i) * cos(0.45j) + c * i + d * j for i in 0:(h - 1), j in 0:(w - 1)]

"""Nested JSON array (rows) -> Matrix."""
tomatrix(rows) = permutedims(reduce(hcat, [Float64.(r) for r in rows]))

relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

"""A small inverse problem on deterministic fields (the one pytorch_reference.py uses)."""
function small_problem(; spw = 2, delta = true, average = false, semi_implicit = true,
                       threaded = false, ntrans = 3)
    H, W = 12, 10
    B0 = testfield(H, W, 20.0, 10.0, 0.5, 0.3)
    targets = [testfield(H, W, 21.0 + k, 9.0 - k, 0.45 + 0.05k, 0.35 - 0.04k) for k in 0:(ntrans - 1)]
    forcings = [sinusoidal_weekly_precip(a) for a in (350.0, 280.0, 410.0)][1:ntrans]
    tr = SiteTrajectory(B0, copy(B0), forcings, targets)
    cfg = SimConfig(steps_per_week = spw, semi_implicit = semi_implicit)
    return InverseProblem([tr], cfg; average = average, delta_loss = delta, threaded = threaded)
end

const FACTORS = [1.3, 0.8, 1.2, 0.9, 1.4, 0.75, 1.1, 0.85, 1.25, 0.7, 1.15]
const P_PERT = RietkerkParams(paramvector(SYNTHETIC_TRUTH) .* FACTORS)

@testset "InverseTuring (Rietkerk backbone)" begin

    # =======================================================================
    @testset "parameters and units" begin
        @test NPARAMS == 11 == length(PARAM_NAMES)
        p = RietkerkParams(collect(1.0:11.0))
        @test paramvector(p) == collect(1.0:11.0)
        @test RietkerkParams(paramdict(p)) == p
        @test_throws ArgumentError RietkerkParams([1.0, 2.0])
        for (i, name) in enumerate(PARAM_NAMES)
            @test getfield(p, i) == getfield(p, Symbol(name))   # struct order == wire order
        end

        # Rietkerk's values on 5 m cells, plant equation slowed 25x.
        t = SYNTHETIC_TRUTH
        @test t.surface_water_diffusion_coeff ≈ 100 / 25
        @test t.soil_water_diffusion_coeff ≈ 0.1 / 25
        @test t.biomass_diffusion_coeff ≈ 0.1 / 25 / 25
        @test t.mortality_rate ≈ 0.25 / 25
        @test t.water_use_efficiency ≈ 10 / 25
        @test (t.seepage_rate, t.infiltration_rate, t.plant_uptake_rate) == (0.2, 0.2, 0.05)
        @test (t.infiltration_half_saturation, t.bare_soil_infiltration, t.uptake_half_saturation) == (5.0, 0.2, 5.0)

        # Real data: 30 m cells, biomass unit 1/15 g/m².
        r = REALDATA_REFERENCE
        @test r.surface_water_diffusion_coeff ≈ 100 / 900
        @test r.plant_uptake_rate ≈ 0.05 / 15
        @test r.water_use_efficiency ≈ 10 / 25 * 15
        @test r.infiltration_half_saturation ≈ 5 * 15
        # to_physical_units undoes pixel size and biomass scale
        phys = to_physical_units(r)
        @test paramvector(phys) ≈ paramvector(rietkerk_reference(pixel_size_m = 1.0, biomass_scale = 1.0))
        # the biomass-unit symmetry: fits under different multipliers agree physically
        @test paramvector(to_physical_units(realdata_reference(750.0); multiplier = 750.0)) ≈ paramvector(phys)

        lo, hi = parameter_bounds(r)
        @test paramvector(lo) ≈ paramvector(r) .* 1e-4
        @test hi.bare_soil_infiltration == 1.0
        @test hi.seepage_rate ≈ r.seepage_rate * 1e4
        llo, lhi = log_bounds(r)
        @test llo ≈ log.(paramvector(lo)) && lhi ≈ log.(paramvector(hi))

        @test isempty(degenerate_parameters(r, r))
        d = paramdict(r)
        d["mortality_rate"] *= 1.05e-4
        @test degenerate_parameters(d, r) == ["mortality_rate"]
        d = paramdict(r)
        d["bare_soil_infiltration"] = 1.0        # W0 = 1 is a legitimate fraction
        @test isempty(degenerate_parameters(d, r))
        d = paramdict(r)
        delete!(d, "plant_uptake_rate")
        @test degenerate_parameters(d, r) == ["plant_uptake_rate"]

        # random starts: within one decade, W0 clamped to 1
        rng = Xoshiro(1)
        for _ in 1:200
            q = randparams(rng, r)
            ratio = paramvector(q) ./ paramvector(r)
            @test all(0.1 - 1e-12 .<= ratio .<= 10 + 1e-12)
            @test q.bare_soil_infiltration <= 1.0
        end
        @test map(x -> 2x, r).seepage_rate == 2 * r.seepage_rate
    end

    # =======================================================================
    @testset "laplacian" begin
        u = fill(3.7, 7, 5)
        @test all(abs.(laplacian(u)) .< 1e-14)
        v = zeros(5, 5)
        v[3, 3] = 1.0
        L = laplacian(v)
        @test L[3, 3] == -4.0
        @test L[2, 3] == L[4, 3] == L[3, 2] == L[3, 4] == 1.0
        r = Float64[j for i in 1:4, j in 1:6]
        Lr = laplacian(r)
        @test all(abs.(Lr[:, 2:5]) .< 1e-14)
        @test all(Lr[:, 1] .≈ 1.0) && all(Lr[:, end] .≈ -1.0)
        w = randn(Xoshiro(3), 9, 11)
        generic = similar(w)
        InverseTuring.laplacian!(generic, view(w, :, :))
        @test laplacian(w) ≈ generic
        @test_throws DimensionMismatch InverseTuring.laplacian!(zeros(3, 3), zeros(4, 4))
    end

    # =======================================================================
    @testset "implicit diffusion" begin
        for (H, W) in ((7, 5), (13, 1), (1, 6), (16, 12), (131, 140), (72, 71))
            u = randn(Xoshiro(H * W), H, W) .+ 3
            for κ in (0.37, 3e-3, 40.0)            # transform, Neumann series, stiff transform
                dense = similar(u)
                implicit_diffusion!(dense, u, κ, MatrixDiffusionSolver(H, W))
                for axes in ((:auto, :auto), (:fftw, :fftw), (:dense, :dense), (:dense, :fftw))
                    fast = similar(u)
                    implicit_diffusion!(fast, u, κ, DiffusionSolver{Float64}(H, W; axes = axes))
                    @test fast ≈ dense rtol = 1e-12
                end
                fast = similar(u)
                implicit_diffusion!(fast, u, κ, DiffusionSolver{Float64}(H, W))
                @test fast .- κ .* laplacian(fast) ≈ u rtol = 1e-12     # it solves (I − κΔ) x = u
                @test sum(fast) ≈ sum(u) rtol = 1e-12                   # zero flux conserves the total
            end
        end
        @test InverseTuring.series_terms(3e-3, Float64, 24) > 0
        @test InverseTuring.series_terms(0.37, Float64, 24) == -1
        @test InverseTuring.series_terms(0.0, Float64, 24) == 0
        @test InverseTuring.fftw_friendly(140) && InverseTuring.fftw_friendly(128)
        @test !InverseTuring.fftw_friendly(131) && !InverseTuring.fftw_friendly(72)
        # κ = 0 is the identity; in-place use is allowed
        u = rand(Xoshiro(2), 6, 4)
        S = DiffusionSolver{Float64}(6, 4)
        x = copy(u)
        implicit_diffusion!(x, x, 0.0, S)
        @test x ≈ u
        x = copy(u)
        implicit_diffusion!(x, x, 0.9, S)
        y = similar(u)
        implicit_diffusion!(y, u, 0.9, S)
        @test x == y
        @test_throws DimensionMismatch implicit_diffusion!(zeros(5, 4), zeros(5, 4), 0.1, S)
        S32 = DiffusionSolver{Float32}(6, 4)
        x32 = similar(u, Float32)
        implicit_diffusion!(x32, Float32.(u), 0.2f0, S32)
        x64 = similar(u)
        implicit_diffusion!(x64, u, 0.2, S)
        @test x32 ≈ x64 rtol = 1e-5

        # Dual numbers: the batched transform and the series match ForwardDiff
        # through the dense solver.
        H, W = 9, 7
        u0 = rand(Xoshiro(5), H, W)
        f(dual, θ) = begin
            D = eltype(θ)
            uu = u0 .* θ[1] .+ θ[2]
            out = similar(uu)
            solver = dual ? diffusion_solver(D, H, W) : MatrixDiffusionSolver(H, W)
            implicit_diffusion!(out, uu, θ[3], solver)
            sum(abs2, out)
        end
        for κ in (0.8, 2e-3)
            θ0 = [1.3, 0.4, κ]
            @test ForwardDiff.gradient(θ -> f(true, θ), θ0) ≈ ForwardDiff.gradient(θ -> f(false, θ), θ0) rtol = 1e-10
        end
        @test diffusion_solver(ForwardDiff.Dual{Nothing,Float64,3}, H, W) isa DiffusionSolver

        # The Enzyme rule against Enzyme through the dense solver (no rule), on both
        # paths, and against finite differences.
        wgt = randn(Xoshiro(6), H, W)
        function g(uu, κ, solver, wgt)
            out = similar(uu)
            implicit_diffusion!(out, uu, κ, solver)
            acc = 0.0
            for i in eachindex(out)
                acc += wgt[i] * out[i]
            end
            return acc
        end
        grad(solver, κ) = (du = zero(u0);
                           dκ = Enzyme.autodiff(Enzyme.Reverse, g, Enzyme.Active,
                                                Enzyme.Duplicated(copy(u0), du), Enzyme.Active(κ),
                                                Enzyme.Const(solver), Enzyme.Const(wgt))[1][2];
                           (du, dκ))
        for κ in (0.8, 2e-3)
            du_fast, dκ_fast = grad(DiffusionSolver{Float64}(H, W), κ)
            du_dense, dκ_dense = grad(MatrixDiffusionSolver(H, W), κ)
            @test du_fast ≈ du_dense rtol = 1e-10
            @test dκ_fast ≈ dκ_dense rtol = 1e-9
            Sfd = MatrixDiffusionSolver(H, W)
            h = 1e-6 * κ
            @test dκ_fast ≈ (g(u0, κ + h, Sfd, wgt) - g(u0, κ - h, Sfd, wgt)) / 2h rtol = 1e-6
        end
    end

    # =======================================================================
    @testset "model step" begin
        # Zero fields and no rain: nothing moves.
        s = SimState(zeros(4, 4), zeros(4, 4), zeros(4, 4))
        ws = workspace(s)
        step!(s, SYNTHETIC_TRUTH, 0.0, 3.5, ws)
        @test all(iszero, s.biomass) && all(iszero, s.surface_water) && all(iszero, s.soil_water)

        # The uniform vegetated equilibrium is an exact fixed point (semi-implicit and explicit).
        p = SYNTHETIC_TRUTH
        R = 1.1
        o, w, b = homogeneous_steady_state(p, R)
        for semi in (true, false)
            s = SimState(fill(o, 5, 6), fill(w, 5, 6), fill(b, 5, 6))
            for _ in 1:20
                step!(s, p, R, semi ? 3.5 : 0.01, workspace(s); semi_implicit = semi)
            end
            @test all(s.surface_water .≈ o) && all(s.soil_water .≈ w) && all(s.biomass .≈ b)
        end

        # Positivity for arbitrary parameters at a large step.
        rng = Xoshiro(4)
        for _ in 1:20
            q = randparams(rng, SYNTHETIC_TRUTH; decades = 3)
            s = SimState(rand(rng, 8, 8) .* 5, rand(rng, 8, 8) .* 5, rand(rng, 8, 8) .* 50)
            ws = workspace(s)
            for _ in 1:30
                step!(s, q, 2.0, 7.0, ws)
            end
            @test all(s.surface_water .>= 0) && all(s.soil_water .>= 0) && all(s.biomass .>= 0)
            @test all(isfinite, s.biomass)
        end

        # Consistency: both schemes converge to the same solution as dt → 0.
        init = SimState(testfield(8, 7, 1.0, 0.3, 0.01, 0.0), testfield(8, 7, 2.0, 0.5, 0.0, 0.02),
                        testfield(8, 7, 20.0, 8.0, 0.3, 0.2))
        run(semi, n) = (s = copystate(init); ws = workspace(s);
                        for _ in 1:n; step!(s, p, 1.2, 7.0 / n, ws; semi_implicit = semi); end; s.biomass)
        e(n) = maximum(abs, run(true, n) .- run(false, n))
        @test e(400) < e(100) / 3          # first order: gap shrinks with dt
        @test e(400) < 1e-3

        # Sequential update: soil water sees this step's surface water.
        s = SimState(zeros(3, 3), zeros(3, 3), fill(1.0, 3, 3))
        step!(s, p, 10.0, 1.0, workspace(s))
        @test all(s.soil_water .> 0)
        @test_throws DimensionMismatch SimState(zeros(2, 2), zeros(3, 3), zeros(2, 2))

        if PYREF !== nothing
            O0 = testfield(12, 10, 1.0, 0.5, 0.02, 0.01)
            W0 = testfield(12, 10, 2.0, 0.8, -0.03, 0.05)
            B0 = testfield(12, 10, 20.0, 10.0, 0.5, 0.3)
            for (key, semi) in (("step_semi_implicit", true), ("step_explicit", false))
                ref = PYREF[key]
                s = SimState(copy(O0), copy(W0), copy(B0))
                step!(s, P_PERT, ref["rate"], ref["dt"], workspace(s); semi_implicit = semi)
                @test s.surface_water ≈ tomatrix(ref["O"]) rtol = 1e-12
                @test s.soil_water ≈ tomatrix(ref["W"]) rtol = 1e-12
                @test s.biomass ≈ tomatrix(ref["B"]) rtol = 1e-12
            end
        end
    end

    # =======================================================================
    @testset "simulate_year" begin
        # Without vegetation and infiltration, surface water collects exactly the
        # annual total, whatever the step.
        dry = RietkerkParams(zeros(NPARAMS))
        dry = RietkerkParams(ntuple(i -> PARAM_NAMES[i] in ("infiltration_half_saturation", "uptake_half_saturation") ? 1.0 : 0.0, NPARAMS))
        weekly = sinusoidal_weekly_precip(300.0)
        for spw in (1, 3, 7)
            s = SimState(zeros(3, 3), zeros(3, 3), zeros(3, 3))
            simulate_year!(s, dry, weekly, SimConfig(steps_per_week = spw))
            @test Statistics.mean(s.surface_water) ≈ 300.0 rtol = 1e-10
        end
        @test_throws ArgumentError simulate_year!(SimState(zeros(2, 2), zeros(2, 2), zeros(2, 2)),
                                                  dry, Float64[], SimConfig(steps_per_week = 1))
        if PYREF !== nothing
            ref = PYREF["year_spw2"]
            s = simstate(testfield(12, 10, 20.0, 10.0, 0.5, 0.3))
            simulate_year!(s, P_PERT, Float64.(PYREF["weekly_350"]), SimConfig(steps_per_week = 2))
            @test s.surface_water ≈ tomatrix(ref["O"]) rtol = 1e-11
            @test s.soil_water ≈ tomatrix(ref["W"]) rtol = 1e-11
            @test s.biomass ≈ tomatrix(ref["B"]) rtol = 1e-11
        end
    end

    # =======================================================================
    @testset "forcing" begin
        w = sinusoidal_weekly_precip(300.0)
        @test length(w) == 52 && annual_total(w) ≈ 300.0 && all(w .>= 0)
        @test argmax(sinusoidal_weekly_precip(300.0; peak_week = 26.0)) == 27
        flat = sinusoidal_weekly_precip(300.0; amplitude_fraction = 0.0)
        @test all(flat .≈ flat[1])
        s = summer_weekly_precip(300.0)
        @test all(s[1:21] .== 0) && all(s[36:end] .== 0)
        @test annual_total(s) ≈ 300.0 * 7 / (365 / 52) rtol = 1e-12
        @test annual_total(summer_weekly_precip(300.0; days_per_week = 7)) ≈ 300.0
        @test_throws ArgumentError summer_weekly_precip(300.0; start_week = 10, end_week = 5)
        if PYREF !== nothing
            @test sinusoidal_weekly_precip(350.0) ≈ Float64.(PYREF["weekly_350"]) rtol = 1e-14
            @test sinusoidal_weekly_precip(420.0; amplitude_fraction = 0.0) ≈
                  Float64.(PYREF["weekly_420_flat"]) rtol = 1e-14
        end
    end

    # =======================================================================
    @testset "loss and gradients" begin
        for delta in (true, false), spw in (2, 3)
            prob = small_problem(; spw = spw, delta = delta)
            L = loss(P_PERT, prob)
            @test isfinite(L) && L > 0
            Le, ge = loss_and_gradient(prob, P_PERT; backend = EnzymeBackend())
            Lf, gf = loss_and_gradient(prob, P_PERT; backend = ForwardDiffBackend())
            @test Le ≈ L rtol = 1e-12
            @test Lf ≈ L rtol = 1e-12
            @test ge ≈ gf rtol = 1e-8
            if PYREF !== nothing
                ref = PYREF["loss_$(delta ? "delta" : "absolute")_spw$spw"]
                @test L ≈ ref["loss_sum"] rtol = 1e-11
                # PyTorch differentiates log θ: ∂L/∂log θ = θ ∂L/∂θ
                glog = ge .* paramvector(P_PERT)
                @test glog ≈ [ref["grad_log_sum"][n] for n in PARAM_NAMES] rtol = 1e-8
            end
        end
        # Finite differences, explicit scheme, averaged loss. Explicit Euler needs
        # dt < 2/spectral_bound ≈ 0.062 day here (D_O = 4 pixel²/day): 150 steps/week.
        @test 7 / 150 < max_stable_dt(SYNTHETIC_TRUTH, 40.0)
        prob = small_problem(; spw = 150, semi_implicit = false, average = true, ntrans = 2)
        _, ge = loss_and_gradient(prob, SYNTHETIC_TRUTH; backend = EnzymeBackend())
        _, gd = loss_and_gradient(prob, SYNTHETIC_TRUTH; backend = FiniteDiffBackend(relstep = 1e-5))
        @test ge ≈ gd rtol = 1e-5
        # ForwardDiff chunk width is a performance knob only.
        prob = small_problem()
        _, g11 = loss_and_gradient(prob, P_PERT; backend = ForwardDiffBackend(chunk = 11))
        _, g4 = loss_and_gradient(prob, P_PERT; backend = ForwardDiffBackend(chunk = 4))
        @test g11 ≈ g4 rtol = 1e-12

        # averaged = summed / transitions
        @test loss(P_PERT, small_problem(average = true)) ≈ loss(P_PERT, small_problem()) / 3
        # zero loss at the truth on noiseless data
        cfg = SimConfig(steps_per_week = 2)
        init = testfield(10, 9, 15.0, 6.0, 0.2, 0.1)
        st = simstate(init)
        prof = sinusoidal_weekly_precip(380.0)
        tgts = map(1:3) do _
            simulate_year!(st, SYNTHETIC_TRUTH, prof, cfg)
            copy(st.biomass)
        end
        clean = InverseProblem([SiteTrajectory(copy(init), copy(init), fill(prof, 3), tgts)], cfg;
                               threaded = false)
        @test loss(SYNTHETIC_TRUTH, clean) < 1e-20
    end

    # =======================================================================
    @testset "threading determinism" begin
        trs = map(1:4) do k
            init = testfield(8, 8, 10.0 + k, 4.0, 0.1, 0.2)
            SiteTrajectory(init, copy(init), [sinusoidal_weekly_precip(300.0 + 20k) for _ in 1:2],
                           [testfield(8, 8, 11.0 + k, 3.0, 0.1, 0.25) for _ in 1:2])
        end
        cfg = SimConfig(steps_per_week = 2)
        serial = InverseProblem(trs, cfg; threaded = false)
        par = InverseProblem(trs, cfg; threaded = true)
        @test loss(P_PERT, serial) === loss(P_PERT, par)
        Ls, gs = loss_and_gradient(serial, P_PERT)
        Lp, gp = loss_and_gradient(par, P_PERT)
        @test Ls === Lp && gs == gp
    end

    # =======================================================================
    @testset "viability" begin
        B0 = testfield(12, 10, 20.0, 10.0, 0.5, 0.3)
        forcings = [sinusoidal_weekly_precip(a) for a in (350.0, 280.0, 410.0)]
        sites = [(B0, forcings)]
        cfg = SimConfig(steps_per_week = 2)
        @test keeps_vegetation(SYNTHETIC_TRUTH, sites, cfg)
        dead = RietkerkParams(paramvector(SYNTHETIC_TRUTH) .* [ones(7); 0.01; ones(3)])
        @test !keeps_vegetation(dead, sites, cfg)
        if PYREF !== nothing
            @test keeps_vegetation(SYNTHETIC_TRUTH, sites, cfg) == PYREF["keeps_vegetation_truth"]
            @test keeps_vegetation(dead, sites, cfg) == PYREF["keeps_vegetation_dead"]
        end
        p, draws = draw_viable_params(Xoshiro(3), SYNTHETIC_TRUTH, sites, cfg)
        @test keeps_vegetation(p, sites, cfg)
        @test draws >= 1
        p2, draws2 = draw_viable_params(Xoshiro(3), SYNTHETIC_TRUTH, sites, cfg)
        @test p2 == p && draws2 == draws                     # reproducible from the seed
    end

    # =======================================================================
    @testset "Adam" begin
        opt = AdamState(3; lr = 0.1, beta1 = 0.9, beta2 = 0.95, eps = 1e-8)
        θ = [1.0, 2.0, 3.0]
        adam_step!(opt, θ, [0.5, -1.0, 0.0])
        @test θ[1] ≈ 0.9 rtol = 1e-6
        @test θ[2] ≈ 2.1 rtol = 1e-6
        @test θ[3] == 3.0
        g = [3.0, 4.0]
        @test clip_global_norm!(g, 1.0) ≈ 5.0
        @test sqrt(sum(abs2, g)) ≈ 1.0 rtol = 1e-6
        opt = AdamState(1; lr = 1.0)
        decay_lr!(opt, 0.5)
        @test opt.lr == 0.5
    end

    # =======================================================================
    @testset "training" begin
        # Noiseless data from the truth; start from a perturbed set and fit.
        cfg = SimConfig(steps_per_week = 2)
        init = testfield(10, 9, 15.0, 6.0, 0.2, 0.1)
        prof = [sinusoidal_weekly_precip(a) for a in (320.0, 400.0, 360.0)]
        st = simstate(init)
        tgts = map(1:3) do k
            simulate_year!(st, SYNTHETIC_TRUTH, prof[k], cfg)
            copy(st.biomass)
        end
        prob = InverseProblem([SiteTrajectory(copy(init), copy(init), prof, tgts)], cfg;
                              average = false, threaded = false)
        start = RietkerkParams(paramvector(SYNTHETIC_TRUTH) .* [1, 1, 1, 1, 1.3, 1, 1, 0.8, 1, 1, 1])
        tc = TrainConfig(SYNTHETIC_TRAIN_CONFIG; epochs = 60, verbose = false)
        res = train(prob, start; cfg = tc, reference = SYNTHETIC_TRUTH)
        @test res.converged
        @test res.final_loss < res.loss_history[1] / 10
        @test length(res.loss_history) == 60
        # snapshots at 0, 10, …, 50 and the final epoch 59, taken after the step
        @test [s["epoch"] for s in res.parameter_history] == [0, 10, 20, 30, 40, 50, 59]
        @test res.parameter_history[1]["mortality_rate"] != start.mortality_rate
        @test res.initial_params == start
        for (i, n) in enumerate(PARAM_NAMES)
            @test res.parameter_history[end][n] == paramvector(res.params)[i]
        end
        lo, hi = parameter_bounds(SYNTHETIC_TRUTH)
        @test all(paramvector(lo) .<= paramvector(res.params) .<= paramvector(hi))

        # Backends agree on the trajectory.
        resf = train(prob, start; cfg = TrainConfig(tc; epochs = 5), reference = SYNTHETIC_TRUTH,
                     backend = ForwardDiffBackend())
        rese = train(prob, start; cfg = TrainConfig(tc; epochs = 5), reference = SYNTHETIC_TRUTH)
        @test paramvector(resf.params) ≈ paramvector(rese.params) rtol = 1e-9

        # Checkpoint/resume continues the same trajectory.
        dir = mktempdir()
        full = train(prob, start; cfg = TrainConfig(tc; epochs = 20), reference = SYNTHETIC_TRUTH)
        ck = joinpath(dir, "ck.json")
        part = train(prob, start; cfg = TrainConfig(tc; epochs = 10), reference = SYNTHETIC_TRUTH,
                     checkpoint_path = ck, checkpoint_every = 10)
        @test isfile(ck)
        resumed = train(prob, start; cfg = TrainConfig(tc; epochs = 20), reference = SYNTHETIC_TRUTH,
                        checkpoint_path = ck, checkpoint_every = 10)
        @test resumed.loss_history ≈ full.loss_history rtol = 1e-12
        @test paramvector(resumed.params) ≈ paramvector(full.params) rtol = 1e-12
        @test [s["epoch"] for s in resumed.parameter_history] == [s["epoch"] for s in full.parameter_history]

        # Random restarts: viable start, reproducible from the seed.
        r1 = train(prob; cfg = TrainConfig(tc; epochs = 3), reference = SYNTHETIC_TRUTH, seed = 7)
        r2 = train(prob; cfg = TrainConfig(tc; epochs = 3), reference = SYNTHETIC_TRUTH, seed = 7)
        @test r1.initial_params == r2.initial_params && r1.init_draws == r2.init_draws
        @test keeps_vegetation(r1.initial_params, viability_sites(prob.trajectories), cfg)
        many = train_many(prob, [7, 8]; cfg = TrainConfig(tc; epochs = 3), reference = SYNTHETIC_TRUTH)
        @test many[1].initial_params == r1.initial_params
        @test paramvector(many[1].params) ≈ paramvector(r1.params) rtol = 1e-12
        @test_throws ArgumentError train_many(prob, [1]; cfg = tc, reference = SYNTHETIC_TRUTH,
                                              parallel = :bogus)
    end

    # =======================================================================
    @testset "linear stability" begin
        p = SYNTHETIC_TRUTH
        # Rietkerk's Turing range T1 = 1.001 < R < T2 = 1.259 mm/day
        @test turing_value(p, 1.1) < 0
        @test turing_value(p, 1.2) < 0
        @test !(turing_value(p, 0.98) < 0)
        @test !(turing_value(p, 1.3) < 0)
        # unchanged by the plant time-scale factor (the reason it is allowed)
        published = rietkerk_reference(pixel_size_m = 5.0, plant_timescale = 1.0)
        for R in (0.98, 1.1, 1.2, 1.3)
            @test (turing_value(published, R) < 0) == (turing_value(p, R) < 0)
        end
        # ~45 m wavelength = ~9 cells of 5 m
        @test 6 < turing_wavelength(p, 1.1) < 14
        o, w, b = homogeneous_steady_state(p, 1.1)
        @test b > 0 && w > 0 && o > 0
        @test homogeneous_steady_state(p, 0.1) === nothing
        @test composite_value(p, 12.0) ≈ (0.05 * 12 / 5) / (0.2 + 0.05 * 12 / 5) * 0.4

        if PYREF !== nothing
            precips = Float64.(PYREF["turing_precips"])
            tv(q) = [turing_value(q, R) for R in precips]
            ref_tv = [x === nothing ? NaN : Float64(x) for x in PYREF["turing_truth"]]
            @test isequal(isnan.(tv(p)), isnan.(ref_tv))
            @test filter(!isnan, tv(p)) ≈ filter(!isnan, ref_tv) rtol = 1e-8
            ref_pert = [x === nothing ? NaN : Float64(x) for x in PYREF["turing_pert"]]
            @test isequal(isnan.(tv(P_PERT)), isnan.(ref_pert))
            @test filter(!isnan, tv(P_PERT)) ≈ filter(!isnan, ref_pert) rtol = 1e-8
            ref_wl = [x === nothing ? NaN : Float64(x) for x in PYREF["turing_wavelength_truth"]]
            wl = [turing_wavelength(p, R) for R in precips]
            @test isequal(isnan.(wl), isnan.(ref_wl))
            @test filter(!isnan, wl) ≈ filter(!isnan, ref_wl) rtol = 1e-8
            @test collect(homogeneous_steady_state(p, 1.1)) ≈ Float64.(PYREF["steady_state_truth_1p1"]) rtol = 1e-14
            @test reaction_jacobian(p, homogeneous_steady_state(p, 1.1)) ≈ tomatrix(PYREF["jacobian_truth_1p1"]) rtol = 1e-14
            @test composite_value(p, 12.0) ≈ PYREF["composite_truth_B12"] rtol = 1e-14
            @test composite_value(REALDATA_REFERENCE, 150.0) ≈ PYREF["composite_realdata_B150"] rtol = 1e-14
        end
    end

    # =======================================================================
    @testset "PyTorch reference: constants" begin
        if PYREF === nothing
            @test_skip "no PyTorch reference"
        else
            @test collect(PARAM_NAMES) == String.(PYREF["param_names"])
            for (key, val) in (("synthetic_truth", SYNTHETIC_TRUTH),
                               ("realdata_reference", REALDATA_REFERENCE),
                               ("realdata_reference_750", realdata_reference(750.0)),
                               ("physical_units_of_realdata_reference", to_physical_units(REALDATA_REFERENCE)))
                @test paramvector(val) == [Float64(PYREF[key][n]) for n in PARAM_NAMES]   # bitwise
            end
            lo, hi = parameter_bounds(REALDATA_REFERENCE)
            for (i, n) in enumerate(PARAM_NAMES)
                @test getfield(lo, i) == PYREF["bounds_realdata"][n][1]
                @test getfield(hi, i) == PYREF["bounds_realdata"][n][2]
            end
            for (k, names) in PYREF["degenerate"]
                snap = Dict{String,Float64}(n => (v === nothing ? NaN : Float64(v))
                                            for (n, v) in PYREF["degenerate_inputs"][k])
                @test degenerate_parameters(snap, REALDATA_REFERENCE) == String.(names)
            end
            u = tomatrix(PYREF["diffusion_input"])
            x = similar(u)
            implicit_diffusion!(x, u, 0.7, DiffusionSolver{Float64}(size(u)...))
            @test x ≈ tomatrix(PYREF["diffusion_out_k0p7"]) rtol = 1e-13
        end
    end

    # =======================================================================
    @testset "analysis" begin
        r = REALDATA_REFERENCE
        good = [paramdict(r) for _ in 1:100]
        short = good[1:50]
        degen = deepcopy(good)
        degen[end]["mortality_rate"] = r.mortality_rate * 1e-4
        kept, dropped = tier1_filter(Dict(1 => good, 2 => short, 3 => degen), r)
        @test kept == [1]
        @test Set(first.(dropped)) == Set([2, 3])
        kept2, dropped2 = drop_degenerate(Dict(1 => good, 2 => short, 3 => degen), r)
        @test kept2 == [1, 2] && first.(dropped2) == [3]

        tbl = agreement_table(Dict("mortality_rate" => fill(r.mortality_rate, 5),
                                   "turing_value" => [-1.0, -1.0]); ground_truth = r)
        @test tbl.parameter == ["mortality_rate", "turing_value"]
        @test tbl.cv[1] == 0.0 && tbl.agreement_score[1] == 1.0
        @test tbl.mape_vs_gt_pct[1] ≈ 0.0 atol = 1e-12
        @test isnan(tbl.ground_truth[2])

        init = fill(10.0, 10, 10) .+ testfield(10, 10, 0.0, 3.0, 0.0, 0.0)
        cfg = SimConfig(steps_per_week = 2)
        levels = [250.0, 350.0, 450.0]
        means, snaps = bifurcation_sweep(SYNTHETIC_TRUTH, init, levels; years = 3, cfg = cfg,
                                         snapshot_at = [350.0])
        @test length(means) == 3 && all(isfinite, means)
        @test issorted(means)
        @test collect(keys(snaps)) == [350.0]
        @test Statistics.mean(snaps[350.0]) ≈ means[2]
    end

    # =======================================================================
    @testset "IO round-trip" begin
        dir = mktempdir()
        res = TrainResult(SYNTHETIC_TRUTH, REALDATA_REFERENCE, [3.0, 2.0, 1.0],
                          [Dict("epoch" => 0.0, "mortality_rate" => 0.3)], 3, 1.0, 0.5, true, 42, 2,
                          [1.0, 0.5, 0.2])
        path = save_run(joinpath(dir, "run_00.json"), res; metadata = Dict("run_id" => 7))
        back = load_run(path)
        @test back["final_loss"] == 1.0 && back["run_id"] == 7 && back["init_draws"] == 2
        @test back["final_params"]["mortality_rate"] == SYNTHETIC_TRUTH.mortality_rate
        csv = write_parameter_table(joinpath(dir, "params.csv"), [0, 1],
                                    [SYNTHETIC_TRUTH, REALDATA_REFERENCE];
                                    extra = Dict("turing_value" => [1.0, 2.0]))
        df = read_parameter_table(csv)
        @test names(df)[1] == "model_id"
        @test params_from_row(df[2, :]) == REALDATA_REFERENCE
    end

    # =======================================================================
    @testset "synthetic experiment" begin
        ex = synthetic_experiment(; preset = :four_site, grid = (16, 16), equilibrium_years = 3,
                                  years = 2, rng = Xoshiro(42), threaded = false)
        @test length(ex.problem) == 4 && ntransitions(ex.problem) == 8
        @test ex.annual_totals ≈ collect(range(15, 27; length = 4)) .* (400 / 21)
        @test !ex.problem.average && ex.problem.cfg.steps_per_week == 2
        @test all(annual_total(p) ≈ t for (p, t) in zip(ex.profiles, ex.annual_totals))
        ex1 = synthetic_experiment(; preset = :one_site, grid = (16, 16), equilibrium_years = 3,
                                   years = 2, rng = Xoshiro(42), threaded = false)
        @test ex1.annual_totals ≈ [16.5 * 400 / 19]
        @test_throws ArgumentError synthetic_experiment(; preset = :nope)
    end

    # =======================================================================
    @testset "continuous model (DifferentialEquations.jl + SciMLSensitivity)" begin
        prob = small_problem(; ntrans = 2)
        ode = with_discretisation(prob, ODEConfig(Tsit5(); abstol = 1e-10, reltol = 1e-10))
        @test ode.cfg isa ODEConfig
        L = loss(P_PERT, ode)
        # The fixed-step scheme is a first-order discretisation of exactly this ODE.
        e(spw) = abs(loss(P_PERT, with_discretisation(prob, SimConfig(steps_per_week = spw))) - L)
        @test e(64) < e(16) < e(4)
        @test e(64) < 2e-3 * L
        @test loss(P_PERT, with_discretisation(prob, ODEConfig(ROCK4(); abstol = 1e-8, reltol = 1e-8))) ≈ L rtol = 1e-6

        # Gradients: continuous adjoint (Enzyme VJPs) = ForwardDiff through the solver = FD.
        La, ga = loss_and_gradient(ode, P_PERT; backend = AdjointODEBackend())
        Lf, gf = loss_and_gradient(ode, P_PERT; backend = ForwardDiffBackend())
        @test La ≈ L rtol = 1e-12
        @test ga ≈ gf rtol = 1e-7
        _, gd = loss_and_gradient(ode, P_PERT; backend = FiniteDiffBackend(relstep = 1e-6))
        @test ga ≈ gd rtol = 1e-4
        ra = SciMLSensitivity.EnzymeVJP(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
        _, gg = loss_and_gradient(ode, P_PERT; backend = AdjointODEBackend(sensealg = GaussAdjoint(autojacvec = ra)))
        @test gg ≈ gf rtol = 1e-7

        # Training runs on the continuous model too, viability screen included.
        res = train(ode; cfg = TrainConfig(SYNTHETIC_TRAIN_CONFIG; epochs = 3, verbose = false),
                    reference = SYNTHETIC_TRUTH, seed = 3, backend = AdjointODEBackend())
        @test res.converged && length(res.loss_history) == 3
        @test keeps_vegetation(res.initial_params, viability_sites(ode.trajectories), ode.cfg)

        # The wrong backend for the discretisation is an error, not a silent fallback.
        @test_throws ArgumentError gradient_cache(ode, EnzymeBackend())
        @test_throws ArgumentError gradient_cache(prob, AdjointODEBackend())

        # The uniform equilibrium is a fixed point of the continuous model too.
        o, w, b = homogeneous_steady_state(SYNTHETIC_TRUTH, 1.1)
        u = InverseTuring.pack_state(fill(o, 5, 6), fill(w, 5, 6), fill(b, 5, 6))
        solve_week!(u, SYNTHETIC_TRUTH, 1.1, ODEConfig(Tsit5(); abstol = 1e-12, reltol = 1e-12))
        @test all(biomass_of(u) .≈ b) && all(view(u, :, :, 1) .≈ o)

        # The Jacobian sparsity pattern contains the true Jacobian.
        H, W = 5, 4
        P = rhs_sparsity(H, W)
        @test size(P) == (3H * W, 3H * W)
        x = rand(Xoshiro(3), H, W, 3) .+ 0.5
        pv = ode_parameters(P_PERT, 1.2)
        J = ForwardDiff.jacobian(x -> (du = similar(x); rietkerk_rhs!(du, x, pv, 0.0); vec(du)), x)
        @test all(iszero, J[iszero.(Matrix(P))])
        @test count(!iszero, J) <= SparseArrays.nnz(P)
    end

    # =======================================================================
    @testset "real data" begin
        if !HAVE_DATA
            @test_skip "data directory not available"
        else
            site = load_site(DATA_DIR, "b"; T = Float32)
            @test site.name == "subsite_b" && length(site) == 10
            @test years(site) == collect(2013:2022)
            @test size(site) == (131, 140)
            for o in site.observations
                @test all(0 .<= o.biomass .<= 1500)
                @test length(o.weekly_precipitation) == 52
            end
            tr = SiteTrajectory(site)
            @test length(tr.targets) == 9 && tr.targets[1] == site.observations[2].biomass

            if PYREF !== nothing && haskey(PYREF, "evaluate_subsite_f_4years_spw4")
                ref = PYREF["evaluate_subsite_f_4years_spw4"]
                f32 = load_site(DATA_DIR, "f"; T = Float32)
                obs = [YearObservation{Float64}(o.year, Float64.(o.biomass), o.precipitation,
                                                o.weekly_precipitation) for o in f32.observations[1:4]]
                @test [o.year for o in obs] == Int.(PYREF["evaluate_years"])
                got = evaluate(REALDATA_REFERENCE, SiteSeries{Float64}("subsite_f", obs),
                               SimConfig(steps_per_week = 4))
                @test got.num_transitions == ref["num_transitions"]
                @test got.mse ≈ ref["mse"] rtol = 1e-10
                @test got.mae ≈ ref["mae"] rtol = 1e-10
                @test got.correlation ≈ ref["correlation"] rtol = 1e-10
            end
        end
    end
end
