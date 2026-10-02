# Enzyme reverse-mode rule for the implicit diffusion solve.
#
# The solve calls FFTW (a C library) or BLAS on the transform path, so the adjoint
# is supplied here rather than derived by Enzyme. For x = A(κ) u with
# A(κ) = (I − κΔ)⁻¹:
#
#   - A is symmetric (Δ is), so ū += A x̄;
#   - dA/dκ = A Δ A, so κ̄ = x̄ᵀ A Δ A u = (A x̄)ᵀ Δ x.
#
# Both only need the operator itself and the 5-point stencil, so one rule covers
# the transform and the Neumann-series path. The only thing carried from the
# forward to the reverse pass is the solution x, and only when κ is active. It is
# kept on a stack owned by the solver rather than in Enzyme's tape: reverse mode
# runs the reverse passes in exactly the opposite order of the forward ones, so the
# buffers are reused from one autodiff call to the next instead of being allocated
# per solve, and the tape is a plain number. The test suite checks the rule against
# Enzyme differentiating the dense-matrix solver (no rule) and against finite
# differences.

# A solver is plans, constants and scratch space: never part of the derivative.
EnzymeRules.inactive_type(::Type{<:DiffusionSolver}) = true

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      ::EnzymeCore.Const{typeof(implicit_diffusion!)},
                                      ::Type{<:EnzymeCore.Const},
                                      out::EnzymeCore.Annotation{<:AbstractMatrix{T}},
                                      u::EnzymeCore.Annotation{<:AbstractMatrix{T}},
                                      κ::EnzymeCore.Annotation{<:Real},
                                      S::EnzymeCore.Const{<:DiffusionSolver{T}}) where {T}
    kv = T(κ.val)
    solver = S.val
    implicit_diffusion!(out.val, u.val, kv, solver)
    if κ isa EnzymeCore.Active
        n = solver.nsaved[] += 1
        n > length(solver.saved) && push!(solver.saved, similar(solver.aux))
        copyto!(solver.saved[n], out.val)
    end
    return EnzymeRules.AugmentedReturn(nothing, nothing, kv)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             ::EnzymeCore.Const{typeof(implicit_diffusion!)},
                             ::Type{<:EnzymeCore.Const}, tape,
                             out::EnzymeCore.Annotation{<:AbstractMatrix{T}},
                             u::EnzymeCore.Annotation{<:AbstractMatrix{T}},
                             κ::EnzymeCore.Annotation{<:Real},
                             S::EnzymeCore.Const{<:DiffusionSolver{T}}) where {T}
    kv = tape
    dκ = zero(kv)
    solver = S.val
    if κ isa EnzymeCore.Active
        x = solver.saved[solver.nsaved[]]
        solver.nsaved[] -= 1
    end
    if !(out isa EnzymeCore.Const)
        x̄ = out.dval
        ax̄ = solver.aux
        implicit_diffusion!(ax̄, x̄, kv, solver)          # A x̄
        if κ isa EnzymeCore.Active
            lap = view(solver.tmp, :, :, 1)               # scratch is free again
            laplacian!(lap, x)
            @inbounds @simd for i in eachindex(ax̄)
                dκ += ax̄[i] * lap[i]
            end
        end
        if u isa EnzymeCore.Const
            fill!(x̄, zero(T))
        elseif u.dval === x̄
            copyto!(x̄, ax̄)                   # in-place solve: the input's adjoint replaces the output's
        else
            fill!(x̄, zero(T))                # `out` was overwritten: its old value does not matter
            u.dval .+= ax̄
        end
    end
    return (nothing, nothing, κ isa EnzymeCore.Active ? dκ : nothing, nothing)
end
