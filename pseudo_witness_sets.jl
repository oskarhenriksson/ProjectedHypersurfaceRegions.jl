using HomotopyContinuation, LinearAlgebra

struct PseudoWitnessSet
    F::System
    k::Int
    L::LinearSubspace
    W::Result
end
degree(PWS::PseudoWitnessSet) = length(PWS.W)


"""
    PseudoWitnessSet(F::System, k::Int; L::Union{LinearSubspace, Nothing} = nothing)

Generates a pseudo witness set for the system `F` under the assumption that the first `k` variables are the downstairs variables.

"""
function PseudoWitnessSet(F::System, k::Int, codimension; L::Union{LinearSubspace, Nothing} = nothing) 
    n = nvariables(F)
    if isnothing(L)
        A = hcat(rand(ComplexF64, codimension, k), zeros(codimension, n-k))
        b = rand(ComplexF64, codimension)
        L = LinearSubspace(A, b)
    else
        @assert codim(L) == codimension "The codimension of the given linear subspace L must match the codimension of the pseudo witness set."
        @assert ambient_dim(L) == n "The ambient dimension of the linear subspace L must match the number of variables in the system F."
    end
    startL = rand_subspace(n; codim = codimension)
    S = solutions(witness_set(F, startL))
    W = HC.solve(F, S, start_subspace = startL, target_subspace = L, intrinsic = true)
    PseudoWitnessSet(F, k, L, W)
end


"""
    lifted_line(point::AbstractVector{<:Number}, direction::AbstractVector{<:Number}, n::Int64)

Lifts the line point+t*direction in R^k to a LinearSubspace in R^k × R^(n-k)
"""
function lifted_line(point::AbstractVector{<:Number}, direction::AbstractVector{<:Number}, n::Int64)
    k = length(point)
    πA = nullspace(direction')'
    A = hcat(πA,zeros(k-1,n-k))
    b = πA * point
    LinearSubspace(A, b)
end


function track_pws_to_lines(
    point::AbstractVector{<:Real},
    directions::AbstractArray{Float64},
    PWS::PseudoWitnessSet,
)
    L = PWS.L
    new_lines = map(bj -> lifted_line(point, bj, ambient_dim(L)), eachcol(directions))
    HC.solve(
        PWS.F,
        PWS.W,
        start_subspace = L,
        target_subspaces = new_lines,
        intrinsic = true,
        transform_result = (r, p) -> solutions(r),
    )
end
