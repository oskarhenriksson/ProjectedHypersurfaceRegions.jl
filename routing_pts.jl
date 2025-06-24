using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra

const HC = HomotopyContinuation

include("grad_r_p.jl")


function routing_points(
    F::System,
    projection_variables::Vector{Variable};
    c::Union{Vector{Float64},Nothing}=nothing,
    B::Union{Matrix{Float64},Nothing}=nothing,
    e::Union{Real,Nothing}=nothing)


    k = length(projection_variables)


    if isnothing(c)
        c = 10 .* randn(k)
    end

    if isnothing(B)
        B = qr(rand(k, k)).Q |> Matrix
    end


    #Getting PWS and degree(PWS)
    all_vars = variables(F)
    x_vars = setdiff(all_vars, projection_variables)
    F_ordered = System(F.expressions, variables=[projection_variables; x_vars])
    u = rand(ComplexF64, k)
    v = rand(ComplexF64, k)
    PWS = PseudoWitnessSet(F_ordered, create_line(u, v, nvariables(F_ordered)))
    d = degree(PWS)

    if isnothing(e)
        e = floor(d / 2) + 1
    end


    #Setting up variables
    m = length(x_vars)
    @var p[1:k] s[1:k, 1:d] u[1:k, 1:d, 1:m]

    # Parameters
    @var Bv[1:k, 1:k]

    # Denominator
    q = 1 + sum((p - c) .* (p - c))
    ∇q = differentiate(q, p)

    # Intersection points should be on the hypersurface   
    evaluated_F = []
    for i = 1:k
        for j = 1:d
            append!(evaluated_F, F_ordered([p + s[i, j] .* Bv[:, i]; u[i, j, :]]))
        end
    end

    # Gradient equations
    gradient_equations = map(1:k) do i
        sum(1 / (-s[i, j]) for j = 1:d) - e * transpose(∇q) * Bv[:, i] / q
    end

    #parameters for monodromy solve
    @var V[1:length(evaluated_F)], W[1:length(gradient_equations)]
    vw0 = zeros(length(V) + length(W))

    #System for monodromy solve
    S = System(
        vcat(evaluated_F - V, gradient_equations - W),
        variables=vcat(s[:], u[:], p),
        parameters=vcat(Bv[:], V[:], W[:]),
    )


    #Defining the symmetric group S_d
    G = SymmetricGroup(d)

    #Labeling of the points, s, intersecting the hypersurface and
    #the solutions, u, of the upstairs system is arbitrary in the d component.
    function relabeling(v::Vector{ComplexF64}, i::Int)

        s_part = reshape(v[1:k*d], k, d)
        u_part = reshape(v[k*d+1:k*d+k*d*m], k, d, m)

        return map(G) do p
            s_part_permuted = copy(s_part)
            u_part_permuted = copy(u_part)
            s_part_permuted[i, :] = s_part[i, p]
            u_part_permuted[i, :, :] = u_part[i, p, :]
            vcat(s_part_permuted[:], u_part_permuted[:], p)
        end

    end


    result = monodromy_solve(S, group_actions=[v -> relabeling(v, i) for i in 1:k])

    # Specialize to our target parameters 
    solns = HC.solve(
        S,
        solutions(result),
        start_parameters=parameters(result),
        target_parameters=[vec(B); vw0],
    )

    p_values = map(sol -> sol[(end-k+1):end], solutions(solns))
    real_p_values = filter(sol -> norm(imag(sol)) < 1e-14, p_values) |> real

    return unique_points(real_p_values)
end


