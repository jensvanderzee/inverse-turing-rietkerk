"""
Shared helpers for the command-line scripts: minimal argument parsing, the
repository paths every script needs, and backend selection.

Kept dependency-free on purpose — the scripts are meant to be readable end to end.
"""

import LinearAlgebra

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

"""Satellite rasters and precipitation CSVs. Read-only."""
const DATA_DIR = joinpath(REPO, "data")

"""
Existing PyTorch results. **Read-only** — scripts may load parameter tables,
metrics and histories from here but never write to it.
"""
const PY_RESULTS = joinpath(REPO, "results")

"""
Where every Julia script writes, mirroring the layout of `results/` (e.g.
`julia/results/real_data_rietkerk/models/...`), so a Julia run never overwrites a
Python one and the two can be compared side by side.
"""
const OUT_ROOT = joinpath(REPO, "julia", "results")

"""
    argval(name, default) -> String

Read `--name value` or `--name=value` from `ARGS`, falling back to `default`.
"""
function argval(name::AbstractString, default)
    flag = "--$name"
    for (i, a) in enumerate(ARGS)
        a == flag && i < length(ARGS) && return ARGS[i + 1]
        startswith(a, flag * "=") && return a[(length(flag) + 2):end]
    end
    return default
end

argint(name, default::Integer) = parse(Int, string(argval(name, default)))
argfloat(name, default::Real) = parse(Float64, string(argval(name, default)))
argflag(name) = "--$name" in ARGS

"""
    arglist(name, default::Vector{String}) -> Vector{String}

Comma-separated list argument, e.g. `--sites b,i,c,e`.
"""
function arglist(name::AbstractString, default::Vector{String})
    raw = argval(name, nothing)
    raw === nothing && return default
    return String.(filter(!isempty, strip.(split(string(raw), ','))))
end

"""
    argrange(name, default) -> UnitRange

`--name START STOP` (Python's `--models 0 3`: models 0, 1, 2), or the default.
"""
function argrange(name::AbstractString, default::UnitRange{Int})
    flag = "--$name"
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i + 2 <= length(ARGS) || error("$flag needs START and STOP")
    return parse(Int, ARGS[i + 1]):(parse(Int, ARGS[i + 2]) - 1)
end

"""Section header, matching the Python console output."""
function banner(title::AbstractString)
    println("\n" * "="^72)
    println(" ", title)
    println("="^72)
    flush(stdout)
end

"""
Print the thread count, and warn when Julia was started single-threaded — the fits
and sweeps thread over sites or restarts. BLAS is pinned to one thread per Julia
thread so the two levels do not fight.
"""
function report_threads()
    n = Threads.nthreads()
    n > 1 && LinearAlgebra.BLAS.set_num_threads(1)
    println("Julia threads: $n  (CPU threads available: $(Sys.CPU_THREADS))")
    n == 1 && @warn "running single-threaded; restart with `julia -t auto` for a large speedup"
    return n
end

"""
    gradient_backend(name) -> AbstractGradientBackend

`enzyme` (default), `forwarddiff` or `finitediff`; `ode-adjoint` needs the problem
built with an `ODEConfig` and `using SciMLSensitivity`.
"""
function gradient_backend(name::AbstractString)
    name == "enzyme" && return EnzymeBackend()
    name == "forwarddiff" && return ForwardDiffBackend()
    name == "finitediff" && return FiniteDiffBackend()
    name == "ode-adjoint" && return AdjointODEBackend()
    error("unknown --backend $name (enzyme, forwarddiff, finitediff, ode-adjoint)")
end
