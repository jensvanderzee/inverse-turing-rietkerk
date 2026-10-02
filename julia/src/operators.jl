"""
    laplacian!(out, u) -> out

Five-point discrete Laplacian with replicate (zero-flux / Neumann) boundaries —
the operator `laplacian` in `rietkerk_model.py` builds with `F.pad(mode="replicate")`
and the kernel `[0 1 0; 1 -4 1; 0 1 0]`.

Grid spacing is folded into the diffusion coefficients (the kernel carries no
`1/h²`), so they are in pixel²/day.

Two implementations: a cache-friendly loop for contiguous CPU arrays, and a
shift-and-accumulate broadcast for every other array type, which is free of scalar
indexing and so works on GPU arrays.
"""
function laplacian! end

"""
Contiguous CPU matrices: a plain `Array`, or a slice like `view(u, :, :, k)` of one
(which is what the ODE backend produces when it splits a packed `H×W×3` state).
Deliberately not `StridedMatrix`, which would also catch GPU arrays.
"""
const ContiguousCPUMatrix = Union{Array{<:Any,2},
                                  SubArray{<:Any,2,<:Array,<:Any,true}}

function laplacian!(out::ContiguousCPUMatrix, u::ContiguousCPUMatrix)
    axes(out) == axes(u) || throw(DimensionMismatch("out and u must have equal axes"))
    H, W = size(u)
    @inbounds for j in 1:W
        jm = j == 1 ? 1 : j - 1
        jp = j == W ? W : j + 1
        # Top row: the upward neighbour replicates the row itself.
        out[1, j] = u[1, j] + u[min(2, H), j] + u[1, jm] + u[1, jp] - 4 * u[1, j]
        @simd for i in 2:(H - 1)
            out[i, j] = u[i - 1, j] + u[i + 1, j] + u[i, jm] + u[i, jp] - 4 * u[i, j]
        end
        if H > 1
            # Bottom row: the downward neighbour replicates the row itself.
            out[H, j] = u[H - 1, j] + u[H, j] + u[H, jm] + u[H, jp] - 4 * u[H, j]
        end
    end
    return out
end

function laplacian!(out::AbstractMatrix, u::AbstractMatrix)
    axes(out) == axes(u) || throw(DimensionMismatch("out and u must have equal axes"))
    H, W = size(u)
    @. out = -4 * u
    @views @. out[2:H, :] += u[1:(H - 1), :]
    @views @. out[1, :] += u[1, :]
    @views @. out[1:(H - 1), :] += u[2:H, :]
    @views @. out[H, :] += u[H, :]
    @views @. out[:, 2:W] += u[:, 1:(W - 1)]
    @views @. out[:, 1] += u[:, 1]
    @views @. out[:, 1:(W - 1)] += u[:, 2:W]
    @views @. out[:, W] += u[:, W]
    return out
end

"""
    laplacian(u)

Allocating form of [`laplacian!`](@ref).
"""
laplacian(u::AbstractMatrix) = laplacian!(similar(u), u)

# ===========================================================================
#  Implicit diffusion: (I − κΔ) x = u, solved exactly
# ===========================================================================
#
# The orthonormal DCT-II basis diagonalises the replicate-boundary 5-point
# Laplacian (eigenvectors cos(πk(2i+1)/2n), eigenvalues −(2 − 2cos(πk/n)) in 1D;
# the 2D operator is the Kronecker sum), so
#
#     x = Cᵀ diag(1 / (1 + κμ)) C u,      μ_kl = (2 − 2cos πk/H) + (2 − 2cos πl/W),
#
# which is `implicit_diffusion` in `rietkerk_model.py` (C applied as two dense
# matrix products). Here C is applied axis by axis in its unnormalised form, the
# DCT-II (FFTW's REDFT10) forward and the DCT-III (REDFT01) back:
# REDFT01(REDFT10(x)) = 2n·x per axis, so x = REDFT01(REDFT10(u) ./ (1 + κμ)) / 4HW.
# Each axis uses FFTW or a dense BLAS product, whichever is faster for its length:
# FFTW is slow for prime lengths (131 takes ~6× longer than 128), and most sites
# here have one.
#
# When 8κ ≪ 1 the Neumann series (I − κΔ)⁻¹ = Σₙ (κΔ)ⁿ reaches machine precision in
# a few 5-point stencil applications, far cheaper than two transforms. At the
# reference parameters that is the case for soil water and biomass (8κ ~ 1e-3 and
# 1e-4); only surface water needs the transform. The series is truncated where the
# remainder is below the rounding error, so both paths give the same operator.

"""Eigenvalue of −Δ (1D, replicate boundaries) for mode `k` of `n`."""
laplacian_eigenvalue(k::Integer, n::Integer) = 2 - 2 * cospi(k / n)

"""Eigenvalues `μ` of −Δ on an `H×W` grid."""
laplacian_spectrum(::Type{T}, H::Integer, W::Integer) where {T} =
    T[laplacian_eigenvalue(k, H) + laplacian_eigenvalue(l, W) for k in 0:(H - 1), l in 0:(W - 1)]

"""
Whether FFTW beats a dense product for a DCT of length `n`: only for long lengths
with no prime factor above 7 (measured on this problem's grid sizes: 128, 135, 140
favour FFTW; 72, 129, 131, 148, 211 the dense product).
"""
function fftw_friendly(n::Integer)
    n >= 96 || return false
    for f in (2, 3, 5, 7)
        while n % f == 0
            n ÷= f
        end
    end
    return n == 1
end

"""Unnormalised DCT-II matrix, FFTW's REDFT10: `M[k, i] = 2cos(πk(2i+1)/2n)`."""
redft10_matrix(::Type{T}, n::Integer) where {T} =
    T[2 * cospi(k * (2i + 1) / (2n)) for k in 0:(n - 1), i in 0:(n - 1)]

"""Unnormalised DCT-III matrix, FFTW's REDFT01: `M[i, k] = k == 0 ? 1 : 2cos(πk(2i+1)/2n)`."""
redft01_matrix(::Type{T}, n::Integer) where {T} =
    T[k == 0 ? one(T) : 2 * cospi(k * (2i + 1) / (2n)) for i in 0:(n - 1), k in 0:(n - 1)]

"""
Dense DCT of one axis, split by symmetry. Mirrored samples `i` and `n−1−i` enter
mode `k` with the same cosine up to the sign `(−1)^k`, so even modes see only the
sums `x_i + x_{n−1−i}` and odd modes only the differences: two half-size matrix
products replace one full one, half the flops of `M * x`. The inverse (DCT-III)
splits the same way. Exact — it is the same matrix, reorganised.
"""
struct DenseDCT{T}
    n::Int
    fe::Matrix{T}     # even modes × (sums, plus the middle sample when n is odd)
    fo::Matrix{T}     # odd modes × differences
    ge::Matrix{T}     # inverse: (pairs, plus middle) × even modes
    go::Matrix{T}     # inverse: pairs × odd modes
    s::Matrix{T}      # scratch, laid out for the axis (see `_axis_dct`)
    d::Matrix{T}
    ye::Matrix{T}
    yo::Matrix{T}
end

function DenseDCT{T}(n::Integer, m::Integer, dim::Integer) where {T}
    h = n ÷ 2
    ne = n - h                          # number of even modes (and of sums incl. the middle)
    c(k, i) = T(2 * cospi(k * (2i + 1) / (2n)))
    fe = T[c(2r, i) for r in 0:(ne - 1), i in 0:(ne - 1)]
    fo = T[c(2r + 1, i) for r in 0:(h - 1), i in 0:(h - 1)]
    ge = T[r == 0 ? one(T) : c(2r, i) for i in 0:(ne - 1), r in 0:(ne - 1)]
    go = T[c(2r + 1, i) for i in 0:(h - 1), r in 0:(h - 1)]
    # Scratch for the reorganised data: rows of an n×m block for axis 1,
    # columns of an m×n block for axis 2.
    sz(k) = dim == 1 ? (k, m) : (m, k)
    return DenseDCT{T}(n, fe, fo, ge, go, zeros(T, sz(ne)...), zeros(T, sz(h)...),
                       zeros(T, sz(ne)...), zeros(T, sz(h)...))
end

# DCT along the rows of an n×m block `X` (axis 1), into `Y`.
function _dense_axis1!(Y::AbstractMatrix, X::AbstractMatrix, ax::DenseDCT, forward::Bool)
    n = ax.n
    h = n ÷ 2
    ne = n - h
    s, d, ye, yo = ax.s, ax.d, ax.ye, ax.yo
    m = size(X, 2)
    if forward
        @inbounds for j in 1:m
            for i in 1:h
                a = X[i, j]
                b = X[n + 1 - i, j]
                s[i, j] = a + b
                d[i, j] = a - b
            end
            isodd(n) && (s[ne, j] = X[ne, j])
        end
        LinearAlgebra.mul!(ye, ax.fe, s)
        LinearAlgebra.mul!(yo, ax.fo, d)
        @inbounds for j in 1:m
            for r in 1:ne
                Y[2r - 1, j] = ye[r, j]
            end
            for r in 1:h
                Y[2r, j] = yo[r, j]
            end
        end
    else
        @inbounds for j in 1:m
            for r in 1:ne
                s[r, j] = X[2r - 1, j]
            end
            for r in 1:h
                d[r, j] = X[2r, j]
            end
        end
        LinearAlgebra.mul!(ye, ax.ge, s)
        LinearAlgebra.mul!(yo, ax.go, d)
        @inbounds for j in 1:m
            for i in 1:h
                a = ye[i, j]
                b = yo[i, j]
                Y[i, j] = a + b
                Y[n + 1 - i, j] = a - b
            end
            isodd(n) && (Y[ne, j] = ye[ne, j])
        end
    end
    return Y
end

# DCT along the columns of an m×n block `X` (axis 2), into `Y`.
function _dense_axis2!(Y::AbstractMatrix, X::AbstractMatrix, ax::DenseDCT, forward::Bool)
    n = ax.n
    h = n ÷ 2
    ne = n - h
    s, d, ye, yo = ax.s, ax.d, ax.ye, ax.yo
    m = size(X, 1)
    if forward
        @inbounds for i in 1:h
            for r in 1:m
                a = X[r, i]
                b = X[r, n + 1 - i]
                s[r, i] = a + b
                d[r, i] = a - b
            end
        end
        isodd(n) && copyto!(view(s, :, ne), view(X, :, ne))
        LinearAlgebra.mul!(ye, s, transpose(ax.fe))
        LinearAlgebra.mul!(yo, d, transpose(ax.fo))
        @inbounds for k in 1:ne
            copyto!(view(Y, :, 2k - 1), view(ye, :, k))
        end
        @inbounds for k in 1:h
            copyto!(view(Y, :, 2k), view(yo, :, k))
        end
    else
        @inbounds for k in 1:ne
            copyto!(view(s, :, k), view(X, :, 2k - 1))
        end
        @inbounds for k in 1:h
            copyto!(view(d, :, k), view(X, :, 2k))
        end
        LinearAlgebra.mul!(ye, s, transpose(ax.ge))
        LinearAlgebra.mul!(yo, d, transpose(ax.go))
        @inbounds for i in 1:h
            for r in 1:m
                a = ye[r, i]
                b = yo[r, i]
                Y[r, i] = a + b
                Y[r, n + 1 - i] = a - b
            end
        end
        isodd(n) && copyto!(view(Y, :, ne), view(ye, :, ne))
    end
    return Y
end

struct FFTWDCT{P,Q}
    fwd::P
    inv::Q
end

"""
    DiffusionSolver{T}(H, W[, planes]; axes = (:auto, :auto), max_terms = 24)

Exact solver for `(I − κΔ) x = u` on an `H×W` grid, for `T = Float64` or
`Float32`. Holds the transforms, the spectrum of −Δ and scratch space, so one
solver must not be used by two tasks at once. `planes > 1` transforms several
fields at once (the value and partials of `ForwardDiff.Dual`s).

`axes` picks the DCT of each axis (`:fftw`, `:dense` or `:auto`, see
[`fftw_friendly`](@ref)); `max_terms` the longest Neumann series used instead of
the transform. Both choices only move rounding error.

Differentiable by Enzyme through a custom rule (`src/enzyme_rules.jl`); `κ` may be
active, which is how the diffusion coefficients get their gradients.
"""
struct DiffusionSolver{T<:Union{Float32,Float64},A1,A2}
    mu::Matrix{T}
    scale::T
    buf::Array{T,3}
    tmp::Array{T,3}
    aux::Matrix{T}
    ax1::A1
    ax2::A2
    max_terms::Int
    # Solutions saved by the Enzyme rule between its forward and reverse passes,
    # last in, first out (see enzyme_rules.jl).
    saved::Vector{Matrix{T}}
    nsaved::Base.RefValue{Int}
end

function _axis_dct(::Type{T}, buf::Array{T,3}, dim::Int, kind::Symbol, flags) where {T}
    H, W, P = size(buf)
    n = size(buf, dim)
    kind === :auto && (kind = fftw_friendly(n) ? :fftw : :dense)
    if kind === :fftw
        fwd = FFTW.plan_r2r!(buf, FFTW.REDFT10, dim; flags = flags)
        inv = FFTW.plan_r2r!(buf, FFTW.REDFT01, dim; flags = flags)
        return FFTWDCT(fwd, inv)
    elseif kind === :dense
        # axis 1 works on the H × (W·P) block, axis 2 plane by plane on H × W
        return dim == 1 ? DenseDCT{T}(H, W * P, 1) : DenseDCT{T}(W, H, 2)
    end
    throw(ArgumentError("axis transform must be :fftw, :dense or :auto, got :$kind"))
end

function DiffusionSolver{T}(H::Integer, W::Integer, planes::Integer = 1;
                            axes::Tuple{Symbol,Symbol} = (:auto, :auto), max_terms::Integer = 24,
                            flags = FFTW.MEASURE) where {T<:Union{Float32,Float64}}
    buf = zeros(T, H, W, planes)
    ax1 = _axis_dct(T, buf, 1, axes[1], flags)
    ax2 = _axis_dct(T, buf, 2, axes[2], flags)
    fill!(buf, zero(T))          # FFTW.MEASURE scribbles on the buffer while planning
    return DiffusionSolver{T,typeof(ax1),typeof(ax2)}(laplacian_spectrum(T, H, W), T(1 / (4 * H * W)),
                                                      buf, zeros(T, H, W, planes), zeros(T, H, W),
                                                      ax1, ax2, max_terms, Matrix{T}[], Ref(0))
end

Base.size(S::DiffusionSolver) = (size(S.buf, 1), size(S.buf, 2))
nplanes(S::DiffusionSolver) = size(S.buf, 3)

# Transform `S.buf` along `dim`, in place.
_apply!(ax::FFTWDCT, S::DiffusionSolver, dim::Int, forward::Bool) =
    (forward ? ax.fwd : ax.inv) * S.buf

function _apply!(ax::DenseDCT, S::DiffusionSolver, dim::Int, forward::Bool)
    H, W, P = size(S.buf)
    if dim == 1
        _dense_axis1!(reshape(S.tmp, H, W * P), reshape(S.buf, H, W * P), ax, forward)
    else
        for q in 1:P
            _dense_axis2!(view(S.tmp, :, :, q), view(S.buf, :, :, q), ax, forward)
        end
    end
    copyto!(S.buf, S.tmp)
    return S.buf
end

dct2!(S::DiffusionSolver) = (_apply!(S.ax1, S, 1, true); _apply!(S.ax2, S, 2, true); S.buf)
idct2!(S::DiffusionSolver) = (_apply!(S.ax1, S, 1, false); _apply!(S.ax2, S, 2, false); S.buf)

"""
Number of Neumann terms that leave a remainder below the rounding error of `T`
(the remainder after `K` terms is at most `(8κ)^(K+1)/(1 − 8κ)` in norm), or `-1`
when that takes more than `max_terms` and the transform is cheaper.
"""
function series_terms(κ::Real, ::Type{T}, max_terms::Integer) where {T}
    r = 8 * abs(κ)
    r == 0 && return 0
    r < 0.5 || return -1
    K = ceil(Int, log(eps(T) * (1 - r) / 2) / log(r)) - 1
    return K <= max_terms ? max(K, 1) : -1
end

"""
Neumann series by Horner's rule, `x = u + κΔ(u + κΔ(u + …))`, with `K` terms.
`y` and `t` are scratch planes; `out` may alias `u` only if `y` does not.
"""
function _series!(out::AbstractMatrix, u::AbstractMatrix, κ, K::Int, y::AbstractMatrix,
                  t::AbstractMatrix)
    copyto!(y, u)
    for _ in 1:K
        laplacian!(t, y)
        @. y = u + κ * t
    end
    copyto!(out, y)
    return nothing
end

"""
    implicit_diffusion!(out, u, κ, solver) -> nothing

Overwrite `out` with the solution `x` of `(I − κΔ) x = u` (`out` and `u` may be the
same array). `κ = dt·D` for a diffusion coefficient `D`.
"""
function implicit_diffusion!(out::AbstractMatrix{T}, u::AbstractMatrix{T}, κ::Real,
                             S::DiffusionSolver{T}) where {T}
    size(u) == size(out) == size(S) ||
        throw(DimensionMismatch("field size $(size(u)) does not match the solver $(size(S))"))
    nplanes(S) == 1 || throw(ArgumentError("this solver transforms $(nplanes(S)) planes; use one plane for plain fields"))
    k = T(κ)
    K = series_terms(k, T, S.max_terms)
    if K >= 0
        _series!(out, u, k, K, view(S.buf, :, :, 1), view(S.tmp, :, :, 1))
        return nothing
    end
    buf = S.buf
    copyto!(buf, u)
    dct2!(S)
    mu = S.mu
    scale = S.scale
    @inbounds @simd for i in eachindex(mu)
        buf[i] = buf[i] / (1 + k * mu[i]) * scale
    end
    idct2!(S)
    copyto!(out, buf)
    return nothing
end

_partials(κ::ForwardDiff.Dual, ::Val{N}) where {N} = ForwardDiff.partials(κ).values
_partials(κ::Real, ::Val{N}) where {N} = ntuple(_ -> zero(κ), N)

"""
`ForwardDiff.Dual` fields: the operator is linear in `u` and smooth in `κ`, so the
value and every partial go through one batched transform (planes of `solver`),

    ŝ = Û_v/(1 + κ_v μ),   ŝ_p = (Û_p − κ_p μ ŝ)/(1 + κ_v μ),

which is far cheaper than pushing `Dual`s through a generic product. The Neumann
path runs on the `Dual`s directly.
"""
function implicit_diffusion!(out::AbstractMatrix{D}, u::AbstractMatrix{D}, κ::Real,
                             S::DiffusionSolver{T}) where {D<:ForwardDiff.Dual,T}
    size(u) == size(out) == size(S) ||
        throw(DimensionMismatch("field size $(size(u)) does not match the solver $(size(S))"))
    N = ForwardDiff.npartials(D)
    nplanes(S) == N + 1 || throw(ArgumentError("solver has $(nplanes(S)) planes, need $(N + 1)"))
    kv = T(ForwardDiff.value(κ))
    K = series_terms(kv, T, S.max_terms)
    if K >= 0
        y = similar(u)
        t = similar(u)
        _series!(out, u, κ, K, y, t)
        return nothing
    end
    buf = S.buf
    H, W = size(S)
    @inbounds for j in 1:W, i in 1:H
        x = u[i, j]
        buf[i, j, 1] = ForwardDiff.value(x)
        ps = ForwardDiff.partials(x)
        for q in 1:N
            buf[i, j, q + 1] = ps[q]
        end
    end
    dct2!(S)
    kp = _partials(κ, Val(N))
    mu = S.mu
    scale = S.scale
    @inbounds for j in 1:W, i in 1:H
        m = mu[i, j]
        f = 1 / (1 + kv * m)
        sv = buf[i, j, 1] * f
        buf[i, j, 1] = sv * scale
        for q in 1:N
            buf[i, j, q + 1] = (buf[i, j, q + 1] - T(kp[q]) * m * sv) * f * scale
        end
    end
    idct2!(S)
    @inbounds for j in 1:W, i in 1:H
        out[i, j] = D(buf[i, j, 1], ForwardDiff.Partials(ntuple(q -> buf[i, j, q + 1], N)))
    end
    return nothing
end

"""
    MatrixDiffusionSolver(H, W)

The algorithm of `rietkerk_model.implicit_diffusion` itself: dense orthonormal DCT
matrices and `x = C_Hᵀ ((C_H u C_Wᵀ) ./ (1 .+ κμ)) C_W`, for any element type. The
reference in the tests, and the fallback for element types FFTW and BLAS cannot
handle.
"""
struct MatrixDiffusionSolver
    ch::Matrix{Float64}
    cw::Matrix{Float64}
    mu::Matrix{Float64}
end

"""Orthonormal DCT-II matrix of size `n`, as `_dct_basis` builds it in Python."""
function dct_matrix(n::Integer)
    c = [cospi(k * (2i + 1) / (2n)) * sqrt(2 / n) for k in 0:(n - 1), i in 0:(n - 1)]
    c[1, :] ./= sqrt(2)
    return c
end

MatrixDiffusionSolver(H::Integer, W::Integer) =
    MatrixDiffusionSolver(dct_matrix(H), dct_matrix(W), laplacian_spectrum(Float64, H, W))

Base.size(S::MatrixDiffusionSolver) = (size(S.ch, 1), size(S.cw, 1))

function implicit_diffusion!(out::AbstractMatrix, u::AbstractMatrix, κ::Real,
                             S::MatrixDiffusionSolver)
    size(u) == size(out) == size(S) ||
        throw(DimensionMismatch("field size $(size(u)) does not match the solver $(size(S))"))
    spectrum = (S.ch * u * transpose(S.cw)) ./ (1 .+ κ .* S.mu)
    out .= transpose(S.ch) * spectrum * S.cw
    return nothing
end

"""
    diffusion_solver(T, H, W) -> solver

The fastest exact solver for fields of element type `T`: [`DiffusionSolver`](@ref)
for `Float64`/`Float32` and for `ForwardDiff.Dual`s of those (one plane per
component), dense matrices otherwise.
"""
diffusion_solver(::Type{T}, H::Integer, W::Integer) where {T<:Union{Float32,Float64}} =
    DiffusionSolver{T}(H, W)
diffusion_solver(::Type{D}, H::Integer, W::Integer) where {D<:ForwardDiff.Dual{<:Any,<:Union{Float32,Float64}}} =
    DiffusionSolver{ForwardDiff.valtype(D)}(H, W, ForwardDiff.npartials(D) + 1)
diffusion_solver(::Type, H::Integer, W::Integer) = MatrixDiffusionSolver(H, W)
