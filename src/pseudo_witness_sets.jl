using HomotopyContinuation, LinearAlgebra

struct PseudoWitnessSet
    F::System
    k::Int
    L::LinearSubspace
    W::Result
end
degree(PWS::PseudoWitnessSet) = length(PWS.W)
ambient_dim(PWS::PseudoWitnessSet) = HC.ambient_dim(PWS.L)

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
    S = solutions(witness_set(F, startL))
    W = HC.solve(F, S, start_subspace = startL, target_subspace = L, intrinsic = true)
    PseudoWitnessSet(F, k, L, W)
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
    LinearSubspace(Complex.(A), Complex.(b))
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
    end

    for (j, K) in enumerate(GC.Ks)
        target_parameters!(GC.tracker, K)
        for (l, w) in enumerate(PWS.W)
            track!(GC.tracker, solution(w), 1)
            GC.line_hypersurface_intersections[j][l] .= solution(GC.tracker)
        end
    end
end
