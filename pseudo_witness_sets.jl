struct PseudoWitnessSet
    F::System
    L::LinearSubspace
    W::Result
end
degree(PWS::PseudoWitnessSet) = length(PWS.W)

function PseudoWitnessSet(F::System, L::LinearSubspace) 
    n = ambient_dim(L)
    startL = rand_subspace(n; codim = codim(L))
    S = solutions(witness_set(F, startL))
    W = HC.solve(F, S, start_subspace = startL, target_subspace = L, intrinsic = true)
    PseudoWitnessSet(F, L, W)
end


function create_line(u::AbstractVector{<:Number}, v::AbstractVector{<:Number}, n::Int64)
    k = length(u)
    #πA = u' - (dot(u, v)/dot(v, v)) * v'
    πA = nullspace(v')'
    A = hcat(πA,zeros(k-1,n-k))
    b = πA * u

    LinearSubspace(A, b)
end

function track_pws_to_lines(
    p::AbstractVector{<:Real},
    B::AbstractArray{Float64},
    PWS::PseudoWitnessSet,
)
    L = PWS.L
    Ks = map(bj -> create_line(p, bj, ambient_dim(L)), eachcol(B))
    HC.solve(
        PWS.F,
        PWS.W,
        start_subspace = L,
        target_subspaces = Ks,
        intrinsic = true,
        transform_result = (r, p) -> solutions(r),
    )
end