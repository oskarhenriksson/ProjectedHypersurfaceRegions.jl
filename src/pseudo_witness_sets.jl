using HomotopyContinuation, LinearAlgebra

struct Line 
    p::Vector
    b::Vector 
end

struct PseudoWitnessSet
    F::System
    k::Int
    L::Line
    W::Vector
    tracker::EndgameTracker
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
    L::Union{Line, Nothing} = nothing,
    start_system::Symbol = :polyhedral,
)
 
    if isnothing(L)
        L = Line(randn(ComplexF64, k), randn(ComplexF64, k))
    end
    
    # Intersect with random linear subspace
    @unique_var t, p[1:k]
    v = variables(F)
    F_L = System([F.expressions; p + t .* L.b - v[1:k]], variables = [v; t], parameters = p)

    # Trace the nonsingular solutions 
    E = HC.solve(F_L; start_system = start_system,
                    target_parameters = L.p)

     # Check for singular solutions
    if nsingular(E) > 0
        @warn "Irreducible component of higher multiplicity detected in the incidence variety."
    end

    # Repopulate the solution set via monodromy (safetey feature if solutions were lost)
    M = monodromy_solve(F_L, solutions(E), L.p)

    # Set up tracker 
    tracker = Tracker(ParameterHomotopy(F_L, L.p, L.p))

    PseudoWitnessSet(F, k, L, solutions(M), EndgameTracker(tracker))
end

function track!(u, PWS::PseudoWitnessSet, p)
    tracker = PWS.tracker
    target_parameters!(tracker, p)
    for (l, w) in enumerate(PWS.W)
            HC.track!(tracker, w, 1)
            u[l] .= solution(tracker)[end]
    end
end




