mutable struct GradientCache{T}
    v0::Vector{T}
    line_hypersurface_intersections::Vector{Vector{T}}
    JsuF::HC.CompiledSystem
    JPF::HC.CompiledSystem
    JBF::HC.CompiledSystem
    HF::HC.CompiledSystem
    JxB::HC.CompiledSystem
    JxP::HC.CompiledSystem
    JPB::HC.CompiledSystem
    S::Vector{T}
    X::Vector{T}
    Uvals::Matrix{T}
    SP::Matrix{T}
    SB::Matrix{T}
    UP::Array{T,3}
    UB::Array{T,3}
    A::Array{T,4}
    rhs1::Matrix{T}
    rhs2::Vector{T}
    rhs3::Vector{T}
    JsuF_vals::Vector{T}
    JPF_vals::Vector{T}
    JBF_vals::Vector{T}
    HF_vals::Vector{T}
    JxB_vals::Vector{T}
    JxP_vals::Vector{T}
    JPB_vals::Vector{T}
    JsuF_temp::Matrix{T}
    JPF_temp::Matrix{T}
    JBF_temp::Matrix{T}
    Jtu_temp::Matrix{T} # Temporary storage for evaluating JsuF
    JsuF_lu::Array{T,3}
    JsuF_ipiv::Matrix{LinearAlgebra.LAPACK.BlasInt}
    JsuF_lu_success::Vector{Bool}
    HF_temp::Array{T, 3} # Temporary storage for evaluating HF
    JxB_temp::Array{T, 3} # Temporary storage for evaluating JxB
    JxP_temp::Array{T, 3} # Temporary storage for evaluating JxP
    JPB_temp::Array{T, 3} # Temporary storage for evaluating JPB
    temp_Hi::Matrix{T}
    temp_Jxpi::Matrix{T}
    temp_Jxbi::Matrix{T}
    temp_Jpbi::Matrix{T}
    ipiv::Vector{LinearAlgebra.LAPACK.BlasInt} # allocation for pivot for lu! in place linear solving
    M::Matrix{T}
    M1::Matrix{T}
    M2::Matrix{T}
    M3::Matrix{T}
    gradient_temp::Vector{T}
    Hess_temp::Matrix{T}
    # A path-local moving copy of the pseudo-witness fibre.  Consecutive calls made by
    # an outer path tracker are close, so continuing this fibre is much cheaper than
    # restarting at the original witness slice for every evaluation.
    fiber_point::Vector{T}
    fiber_solutions::Vector{Vector{T}}
    fiber_scratch::Vector{Vector{T}}
    warm_fiber_tracking::Bool
    fiber_valid::Bool
    fiber_evaluations::Int
    fiber_exact_hits::Int
    fiber_warm_tracks::Int
    fiber_cold_tracks::Int
    fiber_fallbacks::Int
    fiber_failures::Int
    fiber_tracking_ns::UInt64
end

function compute_systems(F, n, k, B)
    @unique_var uval[1:n-k] α[1:k] β[1:k] t
    F_on_line = F([α + (1 / t) * β; uval])
    v = vcat(t, uval)
    vars = vcat(t, uval, α)

    ∇v = map(v) do vi
        HC.ModelKit.differentiate(F_on_line, vi) 
    end
    ∇α = map(α) do αi
        HC.ModelKit.differentiate(F_on_line, αi)
    end

    JsuF_exprs = map(∇v) do ∇vi
        evaluate(∇vi, β => B)
    end
    JPF_exprs = map(∇α) do ∇vi
        evaluate(∇vi, β => B)
    end
    JBF_exprs = map(F_on_line) do f
        evaluate(HC.ModelKit.differentiate(f, β), β => B)
    end

    # Store each derivative block in one compiled system to avoid many tiny system evaluations.
    JsuF = CompiledSystem(System(reduce(vcat, JsuF_exprs), variables = vars))
    JPF = CompiledSystem(System(reduce(vcat, JPF_exprs), variables = vars))
    JBF = CompiledSystem(System(reduce(vcat, JBF_exprs), variables = vars))

    function J(x) 
        map(Iterators.product(∇v, x)) do (∇vi, xj)
            evaluate(HC.ModelKit.differentiate(∇vi, xj), β => B)
        end
    end

    HF_exprs = J(v)
    JxB_exprs = J(β)
    JxP_exprs = J(α)
    JPB_exprs = map(Iterators.product(∇α, β)) do (∇αi, βj)
        evaluate(HC.ModelKit.differentiate(∇αi, βj), β => B)
    end
    HF = CompiledSystem(System(reduce(vcat, vec(HF_exprs)), variables = vars))
    JxB = CompiledSystem(System(reduce(vcat, vec(JxB_exprs)), variables = vars))
    JxP = CompiledSystem(System(reduce(vcat, vec(JxP_exprs)), variables = vars))
    JPB = CompiledSystem(System(reduce(vcat, vec(JPB_exprs)), variables = vars))

    return JsuF, JPF, JBF, HF, JxB, JxP, JPB

end

function GradientCache(PWS)
    d = degree(PWS)
    k = n_projection_variables(PWS)
    F = PWS.F
    L = PWS.L
    N, n = size(F)

    @assert N == n-k+1 "Unexpected length of system"

    # The restricted tracker stores [t; w], so each tracked point has length n - k + 1.
    line_hypersurface_intersections = [zeros(ComplexF64, n - k + 1) for _ in 1:d]
  
    @unique_var t, p[1:k]

    S = zeros(ComplexF64, d)
    X = zeros(ComplexF64, k)
    Uvals = zeros(ComplexF64, n - k, d)
    SP = zeros(ComplexF64, d, k)
    SB = zeros(ComplexF64, d, k)
    UP = zeros(ComplexF64, d, n - k, k)
    UB = zeros(ComplexF64, d, n - k, k)
    A = zeros(ComplexF64, d, N, k, k) 

    JsuF, JPF, JBF, HF, JxB, JxP, JPB = compute_systems(F, n, k, L.direction)


    # 
    rhs1 = zeros(ComplexF64, N, 2*k)  
    rhs2 = zeros(ComplexF64, N)  
    rhs3 = zeros(ComplexF64, k)  
    JsuF_vals = zeros(ComplexF64, N * N)
    JPF_vals = zeros(ComplexF64, N * k)
    JBF_vals = zeros(ComplexF64, N * k)
    HF_vals = zeros(ComplexF64, N * N * N)
    JxB_vals = zeros(ComplexF64, N * N * k)
    JxP_vals = zeros(ComplexF64, N * N * k)
    JPB_vals = zeros(ComplexF64, N * k * k)

    JsuF_temp = zeros(ComplexF64, N, 1+n-k)
    JPF_temp = zeros(ComplexF64, N, k)
    JBF_temp = zeros(ComplexF64, k, N)
    Jtu_temp = zeros(ComplexF64, N, 1+n-k) # TODO: Maybe can reuse Jsu_temp....
    JsuF_lu = zeros(ComplexF64, d, N, N)
    JsuF_ipiv = Matrix{LinearAlgebra.LAPACK.BlasInt}(undef, d, N)
    JsuF_lu_success = zeros(Bool, d)
    HF_temp = zeros(ComplexF64, N, N, N)
    JxB_temp = zeros(ComplexF64, N, N, k)
    JxP_temp = zeros(ComplexF64, N, N, k)
    JPB_temp = zeros(ComplexF64, N, k, k)
    temp_Hi = zeros(ComplexF64, N, N)
    temp_Jxpi = zeros(ComplexF64, N, k)
    temp_Jxbi = zeros(ComplexF64, N, k)
    temp_Jpbi = zeros(ComplexF64, k, k)

    ipiv = Vector{LinearAlgebra.LAPACK.BlasInt}(undef, min(size(JsuF_temp,1), size(JsuF_temp,2)))

    M = zeros(ComplexF64, k, k)
    M1 = zeros(ComplexF64, k, n-k+1)
    M2 = zeros(ComplexF64, n-k+1, k)
    M3 = zeros(ComplexF64, k, n-k+1)

    gradient_temp = zeros(ComplexF64, k)
    Hess_temp = zeros(ComplexF64, k, k)

    fiber_point = zeros(ComplexF64, k)
    fiber_solutions = [copy(z) for z in PWS.tZ]
    fiber_scratch = [similar(z) for z in PWS.tZ]

    v0 = randn(ComplexF64, n+1)

    GradientCache{ComplexF64}(v0, 
                    line_hypersurface_intersections,
                    JsuF,
                    JPF,
                    JBF,
                    HF,
                    JxB,
                    JxP,
                    JPB,
                    S, 
                    X, 
                    Uvals, 
                    SP, 
                    SB, 
                    UP, 
                    UB, 
                    A, 
                    rhs1, 
                    rhs2, 
                    rhs3,
                    JsuF_vals,
                    JPF_vals,
                    JBF_vals,
                    HF_vals,
                    JxB_vals,
                    JxP_vals,
                    JPB_vals,
                    JsuF_temp, 
                    JPF_temp, 
                    JBF_temp, 
                    Jtu_temp, 
                    JsuF_lu,
                    JsuF_ipiv,
                    JsuF_lu_success,
                    HF_temp, 
                    JxB_temp, 
                    JxP_temp, 
                    JPB_temp, 
                    temp_Hi,
                    temp_Jxpi,
                    temp_Jxbi,
                    temp_Jpbi,
                    ipiv,
                    M, 
                    M1, 
                    M2, 
                    M3, 
                    gradient_temp, 
                    Hess_temp,
                    fiber_point,
                    fiber_solutions,
                    fiber_scratch,
                    true,
                    false,
                    0, 0, 0, 0, 0, 0,
                    UInt64(0)
                )
end

@inline function _same_fiber_point(a, b)
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        a[i] == b[i] || return false
    end
    true
end

function _track_fiber_from!(dest, PWS::PseudoWitnessSet, starts, p_start, p_target)
    tracker = PWS.tracker
    start_parameters!(tracker, p_start)
    target_parameters!(tracker, p_target)
    succeeded = true
    for (i, start) in enumerate(starts)
        code = HC.track!(tracker, start, 1)
        copyto!(dest[i], tracker.tracker.state.x)
        ok = HC.is_success(code) && all(isfinite, dest[i])
        PWS.track_report[i] = ok
        succeeded &= ok
    end
    succeeded
end

"""Track the pseudo-witness fibre to `p`, reusing the most recent fibre when possible.

Updates are transactional: a failed warm track is discarded and retried from the
original witness slice.  An incomplete fibre is never used for differentiation.
"""
function track!(GC::GradientCache, PWS::PseudoWitnessSet, p)
    GC.fiber_evaluations += 1

    if GC.warm_fiber_tracking && GC.fiber_valid && _same_fiber_point(GC.fiber_point, p)
        GC.fiber_exact_hits += 1
        for i in eachindex(GC.line_hypersurface_intersections)
            copyto!(GC.line_hypersurface_intersections[i], GC.fiber_solutions[i])
            PWS.track_report[i] = true
        end
        get_s_and_Uvals!(GC.Uvals, GC.S, GC, PWS)
        return nothing
    end

    t0 = time_ns()
    success = false
    if GC.warm_fiber_tracking && GC.fiber_valid
        GC.fiber_warm_tracks += 1
        success = _track_fiber_from!(
            GC.fiber_scratch, PWS, GC.fiber_solutions, GC.fiber_point, p,
        )
        if !success
            GC.fiber_fallbacks += 1
        end
    end

    if !success
        GC.fiber_cold_tracks += 1
        success = _track_fiber_from!(GC.fiber_scratch, PWS, PWS.tZ, PWS.L.point, p)
    end
    GC.fiber_tracking_ns += UInt64(time_ns() - t0)

    if !success
        GC.fiber_failures += 1
        GC.fiber_valid = false
        error("Failed to track the complete pseudo-witness fibre to the evaluation point.")
    end

    copyto!(GC.fiber_point, p)
    for i in eachindex(GC.fiber_solutions)
        copyto!(GC.fiber_solutions[i], GC.fiber_scratch[i])
        copyto!(GC.line_hypersurface_intersections[i], GC.fiber_scratch[i])
        PWS.track_report[i] = true
    end
    GC.fiber_valid = GC.warm_fiber_tracking
    get_s_and_Uvals!(GC.Uvals, GC.S, GC, PWS)
    nothing
end

function set_warm_fiber_tracking!(GC::GradientCache, enabled::Bool)
    GC.warm_fiber_tracking = enabled
    GC.fiber_valid = false
    GC
end

function reset_fiber_cache!(GC::GradientCache)
    GC.fiber_valid = false
    GC.fiber_evaluations = 0
    GC.fiber_exact_hits = 0
    GC.fiber_warm_tracks = 0
    GC.fiber_cold_tracks = 0
    GC.fiber_fallbacks = 0
    GC.fiber_failures = 0
    GC.fiber_tracking_ns = UInt64(0)
    GC
end

fiber_tracking_stats(GC::GradientCache) = (
    enabled = GC.warm_fiber_tracking,
    evaluations = GC.fiber_evaluations,
    exact_hits = GC.fiber_exact_hits,
    warm_tracks = GC.fiber_warm_tracks,
    cold_tracks = GC.fiber_cold_tracks,
    fallbacks = GC.fiber_fallbacks,
    failures = GC.fiber_failures,
    tracking_seconds = Float64(GC.fiber_tracking_ns) / 1e9,
)
