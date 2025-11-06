
# copied and adapted from https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl/blob/main/src/homotopies/parameter_homotopy.jl

struct RoutingPointsHomotopy <: AbstractHomotopy
    r::RoutingGradient
    p::Vector{ComplexF64}
    q::Vector{ComplexF64}
    #cache
    t_cache::Base.RefValue{ComplexF64}
    pt::Vector{ComplexF64}
    taylor_pt::TaylorVector{2,ComplexF64}
end

function RoutingPointsHomotopy(r::RoutingGradient, p::AbstractVector, q::AbstractVector)
    @assert length(p) == length(q) == size(r)[1]

    p̂ = Vector{ComplexF64}(p)
    q̂ = Vector{ComplexF64}(q)
    taylor_pt = TaylorVector{2}(ComplexF64, length(q))
    pt = copy(p̂)

    RoutingPointsHomotopy(r, p̂, q̂, Ref(complex(NaN)), pt, taylor_pt)
end

Base.size(H::RoutingPointsHomotopy) = size(H.r)

import HomotopyContinuation.start_parameters!
import HomotopyContinuation.target_parameters!
import HomotopyContinuation.parameters!

function start_parameters!(H::RoutingPointsHomotopy, p)
    H.p .= p
    # void cache
    H.t_cache[] = NaN
    H
end
function target_parameters!(H::RoutingPointsHomotopy, q)
    H.q .= q
    H.t_cache[] = NaN
    H
end
function parameters!(H::RoutingPointsHomotopy, p, q)
    start_parameters!(H, p)
    target_parameters!(H, q)
end
function HomotopyContinuation.parameters!(H::RoutingPointsHomotopy, p, q)
    H.p .= p
    H.q .= q
    H.t_cache[] = NaN
    H
end

function tp!(H::RoutingPointsHomotopy, t::Union{ComplexF64,Float64})
    t == H.t_cache[] && return H.taylor_pt

    if imag(t) == 0
        let t = real(t)
            @inbounds for i = 1:length(H.taylor_pt)
                ptᵢ = t * H.p[i] + (1.0 - t) * H.q[i]
                H.pt[i] = ptᵢ
                H.taylor_pt[i] = (ptᵢ, H.p[i] - H.q[i])
            end
        end
    else
        @inbounds for i = 1:length(H.taylor_pt)
            ptᵢ = t * H.p[i] + (1.0 - t) * H.q[i]
            H.pt[i] = ptᵢ
            H.taylor_pt[i] = (ptᵢ, H.p[i] - H.q[i])
        end
    end
    H.t_cache[] = t

    H.taylor_pt
end

function ModelKit.evaluate!(u, H::RoutingPointsHomotopy, x, t)
    tp!(H, t)
    evaluate!(u, H.r, x, H.pt)
end

function ModelKit.evaluate_and_jacobian!(u, U, H::RoutingPointsHomotopy, x, t)
    tp!(H, t)
    evaluate_and_jacobian!(u, U, H.r, x, H.pt)
end

function ModelKit.taylor!(u, v::Val, H::RoutingPointsHomotopy, tx, t)
    taylor!(u, v, H.r, tx, tp!(H, t))
    u
end

import HomotopyContinuation: MonodromyOptions, UniquePoints, EndgameTracker
function critical_points(r::RoutingGradient,
                            S0::Union{AbstractVector{<:AbstractVector{<:Number}}, Nothing} = nothing,
                            rhs0::Union{AbstractVector{<:Number}, Nothing} = nothing;
                            verbose = true,
                            num_gradient_flow_starts = 100,
                            monodromy_at_zero = false,
                            options = MonodromyOptions(parameter_sampler = p -> 10 .* randn(ComplexF64, length(p))),  
                            seed = rand(UInt32))
    k = size(r, 2) # number of variables
    p1 = zeros(k)
    q1 = randn(k)
    H = RoutingPointsHomotopy(r, p1, q1)

    ### Use monodromy to the system ∇r = rhs0 where we view the right-hand side are the parameters of the system
    egtracker = EndgameTracker(H) # we want to add options later
    trackers = [egtracker]
    x₀ = zeros(ComplexF64, size(H, k))

    unique_points = UniquePoints(
        x₀,
        1;
    )

    trace = zeros(ComplexF64, length(x₀) + 1, 3)
    P = Vector{ComplexF64}
    MS = HomotopyContinuation.MonodromySolver(
        trackers,
        HomotopyContinuation.MonodromyLoop{P}[],
        unique_points,
        ReentrantLock(),
        options,
        HomotopyContinuation.MonodromyStatistics(),
        trace,
        ReentrantLock(),
    )

    #### set up start pair
    if !monodromy_at_zero
        if isnothing(rhs0) || isnothing(S0)
            s0 = randn(ComplexF64, k)
            rhs0 = evaluate(r, s0)
            S0 = [s0]
        end
    else
        rhs0 = zeros(k)
        if isnothing(S0)
            S0 = Vector{ComplexF64}[]
        end
    end

    ### Expand S0: find solutions of ∇r=0 through gradient descent and trace to ∇r=rhs0
    new_pts = Vector{ComplexF64}[]
    g(x, param, t) = real(evaluate(r, x))
    tspan = (0.0, 1e4)

    if num_gradient_flow_starts>0
        verbose && println("Expanding the set of start solutions via gradient flow...")
        for _ in 1:num_gradient_flow_starts
            start_point = randn(k)
            v = randn(k); v = v / norm(v);
            prob = ODEProblem(g, start_point+0.001*v, tspan)
            sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
            convergence_point = last(sol.u)
            improved_point = newton(r, convergence_point) |> solution
            push!(new_pts, improved_point)
        end
        new_pts = HC.unique_points(new_pts)
        verbose && println("Found $(length(new_pts)) routing points via gradient flow.")
        if !monodromy_at_zero
            start_parameters!(H, zeros(ComplexF64, length(rhs0)));
            target_parameters!(H, rhs0);
            S0_new_sols = HC.solve(H, new_pts) |> solutions
            number_of_old_sols = length(S0)
            S0 = HC.unique_points([S0; S0_new_sols])
            verbose && println("Traced to $(length(S0)-number_of_old_sols) additional start solutions for the monodromy.")
        else
            S0 = [S0; new_pts]
        end
    end

    ### Monodromy
    mon_result = monodromy_solve(
        MS,
        S0,
        rhs0,
        rand(UInt32);
    )

    ### Trace to ∇r=0
    if !monodromy_at_zero
        intermediate_rhs = randn(ComplexF64, length(rhs0))
        start_parameters!(H, rhs0)
        target_parameters!(H, intermediate_rhs)
        result_intermediate = HomotopyContinuation.solve(H, solutions(mon_result))
        start_parameters!(H, intermediate_rhs)
        target_parameters!(H, zeros(ComplexF64, length(rhs0)))
        result = HomotopyContinuation.solve(H, result_intermediate)
        routing_points = real_solutions(result)

        # Make sure none of the routing points found via gradient flow are lost
        if num_gradient_flow_starts>0
            routing_points = HC.unique_points([routing_points; real.(new_pts)])
        end
        return routing_points, result, mon_result
    else
        routing_points = real_solutions(results(mon_result))
        return routing_points, mon_result, mon_result
    end

    return routing_points, result, mon_result
end