using HomotopyContinuation, LinearAlgebra

struct PseudoWitnessSet
    F::System
    k::Int
    L::LinearSubspace
    W::Vector
end
degree(PWS::PseudoWitnessSet) = length(PWS.W)
ambient_dim(PWS::PseudoWitnessSet) = HC.ambient_dim(PWS.L)
n_projection_variables(PWS::PseudoWitnessSet) = PWS.k
system(PWS::PseudoWitnessSet) = PWS.F

@doc raw"""
    PseudoWitnessSet(F::System, k::Int; linear_subspace_codim::Int, L::LinearSubspace)

Generates a pseudo witness set for the image of the variety $V(F)\subseteq\mathbb{C}^n$
for a system $F\in\mathbb{C}[x_1,\ldots,x_n]^r$ under the projection $\pi\colon\mathbb{C}^k\times\mathbb{C}^{n-k}\to\mathbb{C}^k$.
     

Optional inputs:

- `linear_subspace_codim`: The codimension of the linear space used for the witness set. Defaults to `n - length(F)`.
- `L`: The linear space used for the witness set. Should be the preimage under $\pi$ of a linear subspace in $\mathbb{C}^k$

"""
function PseudoWitnessSet(
    F::System,
    k::Int;
    linear_subspace_codim::Union{Int,Nothing} = nothing,
    L::Union{LinearSubspace,Nothing} = nothing,
    start_system::Symbol = :polyhedral
)
    n = nvariables(F)
    if isnothing(linear_subspace_codim)
        linear_subspace_codim = n - length(F)
    end
    if isnothing(L)
        A = hcat(
            rand(ComplexF64, linear_subspace_codim, k),
            zeros(linear_subspace_codim, n - k),
        )
        b = rand(ComplexF64, linear_subspace_codim)
        L = LinearSubspace(A, b)
    else
        @assert ambient_dim(L) == n "The ambient dimension of the linear subspace L must match the number of variables in the system F."
    end
    startL = rand_subspace(n; codim = linear_subspace_codim)

    W = witness_set(F, startL; start_system = start_system)
    E = HC.solve(F, results(W), start_subspace = startL, target_subspace = L, intrinsic = true)
    M = monodromy_solve(F, solutions(E), L)

    PseudoWitnessSet(F, k, L, solutions(M))
end


@doc raw"""
    lifted_line(point::AbstractVector, direction::AbstractVector, n::Int64)

Lifts the line `point+t*direction` in $\mathbb{C}^k$ to a LinearSubspace in $\mathbb{C}^k\timess \mathbb{C}^{n-k}$.
"""
function lifted_line( 
    point::AbstractVector,
    direction::AbstractVector,
    n::Int64,
)
    k = length(point)
    πA = transpose(nullspace(direction'))
    A = hcat(πA, zeros(k - 1, n - k))
    b = πA * point
    LinearSubspace(ComplexF64.(A), ComplexF64.(b))
end


function track_pws_to_lines!(
    GC,
    point::AbstractVector,
    directions::AbstractArray{Float64},
    PWS::PseudoWitnessSet,
)
    n = ambient_dim(PWS)
    for (j, bj) in enumerate(eachcol(directions))
        GC.Ks[j] = lifted_line(point, bj, n)
        target_parameters!(GC.tracker, GC.Ks[j])
        for (l, w) in enumerate(PWS.W)
            track!(GC.tracker, w, 1)
            GC.line_hypersurface_intersections[j][l] .= solution(GC.tracker)
        end
    end
end


"""track_pws_to_line!(GC, point, direction, PWS)

Tracks the pseudo-witness set `PWS` to a single lifted line defined by
`point` and `direction`. Results are stored in `GC.Ks[1]` and
`GC.line_hypersurface_intersections[1]`, matching the shape used by
`GradientCache(...; single_slice=true)`.
"""
function track_pws_to_line!(
    GC,
    point::AbstractVector,
    direction::AbstractVector{Float64},
    PWS::PseudoWitnessSet,
)
    n = ambient_dim(PWS)
    GC.Ks[1] = lifted_line(point, direction, n)
    target_parameters!(GC.tracker, GC.Ks[1])
    for (l, w) in enumerate(PWS.W)
        track!(GC.tracker, w, 1)
        GC.line_hypersurface_intersections[1][l] .= solution(GC.tracker)
        GC.track_report[l] = all(!isnan, GC.line_hypersurface_intersections[1][l]) # note if the track was successful or not
        if GC.track_report[l] == false
            @warn "Track $l failed. This point will be ignored in gradient and Hessian computations."
        end
    end
end
