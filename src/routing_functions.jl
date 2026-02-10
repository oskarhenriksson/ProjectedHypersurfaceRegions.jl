export RoutingFunction, evaluate, evaluate_and_jacobian, gradient, hessian, denominator_exponent, RoutingGradient

struct RoutingFunction{TQ,TP,TC} <: HC.AbstractSystem
    h::ProjectedHypersurface{TC} 
    projection_vars::Vector{HC.Variable}
    e::Int
    c::Vector
    G::Union{MixedSystem, Nothing}
    ∇logprodg::Union{TP, Nothing}
    q::Expression
    ∇logqe::Union{TQ, Nothing}
end
function RoutingFunction(
    h::ProjectedHypersurface;
    e::Union{Int,Nothing} = nothing,
    c::Union{Vector,Nothing} = nothing,
    g::Union{Vector{Expression},Vector{Variable},Nothing} = nothing
)
    
    k = nvariables(h)
    projection_vars = h.projection_vars

    if isnothing(g) || length(g) == 0
        ∇logprodg = nothing
        g_degree = 0
        G = nothing
    else
        g = Expression.(g)
        @assert ModelKit.variables(g) ⊆ projection_vars "Variables in g must match projection_vars"
        G = System(g, variables=projection_vars) |> fixed
        ∇logprodg = System(sum([differentiate(log(gi), projection_vars) for gi in g]), variables=projection_vars) |> fixed
        g_degree = sum(HC.degree.(g))
    end

    if isnothing(e)
        e = div(degree(h) + g_degree, 2) + 1
    end

    if isnothing(c)
        c = randn(k)
    end

    q = 1 + sum((projection_vars - c) .* (projection_vars - c))
    ∇logqe = System(differentiate(-e * log(q), projection_vars), variables = projection_vars) |> fixed

    RoutingFunction{typeof(∇logqe), typeof(∇logprodg), typeof(h.GC)}(h, projection_vars, e, c, G, ∇logprodg, q, ∇logqe)
end

denominator_exponent(r::RoutingFunction) = r.e
ModelKit.variables(r::RoutingFunction) = r.projection_vars
ModelKit.nvariables(r::RoutingFunction) = length(r.projection_vars)

function Base.show(io::IO, r::RoutingFunction)
    print(io, "RoutingFunction(degree=$(degree(r.h)), vars=$(nvariables(r))")
end


function ModelKit.evaluate(r::RoutingFunction, x, p = nothing)
    h = r.h
    e, c = r.e, r.c
    G = r.G

    if !isnothing(G)
        u = sum(log(abs(gi)) for gi in G(x)) - e * log(1 + sum((x - c) .* (x - c)))
    else
        u = - e * log(1 + sum((x - c) .* (x - c)))
    end

    u += h(x)

    u
end

(r::RoutingFunction{TQ,TP,TC})(x) where {TQ,TP,TC} = evaluate(r, x)


function gradient!(u, r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC}
    
    h, ∇logqe, ∇logprodg = r.h, r.∇logqe, r.∇logprodg

    GC = h.GC
    gradient_temp = GC.gradient_temp

    # Denomiator
    evaluate!(u, ∇logqe, x)

    # Known numerator
    if !isnothing(∇logprodg)
        evaluate!(gradient_temp, ∇logprodg, x)
        u .+= gradient_temp
    end

    # Projected hypersurface
    gradient!(gradient_temp, h, x)
    u .+= gradient_temp

    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end

    nothing
end
function gradient(r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC}
    k = nvariables(r)
    u = zeros(ComplexF64, k)
    gradient!(u, r, x, p)
    u
end


function gradient_and_hessian!(u, U, r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC}

    h, ∇logqe, ∇logprodg = r.h, r.∇logqe, r.∇logprodg

    GC = h.GC

    gradient_temp = GC.gradient_temp
    Hess_temp = GC.Hess_temp

    # Denominator
    evaluate_and_jacobian!(u, U, ∇logqe, x)

    # Known numberator
    if !isnothing(∇logprodg)
        evaluate_and_jacobian!(gradient_temp, Hess_temp, ∇logprodg, x)
        u .+= gradient_temp
        U .+= Hess_temp
    end

    # Projected hypersurface
    gradient_and_hessian!(gradient_temp, Hess_temp, h, x)
    u .+= gradient_temp
    U .+= Hess_temp

    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end

    nothing
end
function gradient_and_hessian(r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC}
    k = nvariables(r)
    u = zeros(ComplexF64, k)
    U = zeros(ComplexF64, k, k)
    gradient_and_hessian!(u, U, r, x, p)
    u, U
end
hessian(r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC} =
    gradient_and_hessian(r, x, p)[2]

#########

struct RoutingGradient <: HC.AbstractSystem
    r::RoutingFunction
end

function Base.show(io::IO, ∇r::RoutingGradient)
    print(io, "RoutingGradient(vars=$(nvariables(∇r.r))×$(nvariables(∇r.r))")
end

import Base.size
function Base.size(∇r::RoutingGradient)
    k = nvariables(∇r.r)
    (k, k)
end
ModelKit.variables(∇r::RoutingGradient) = ∇r.r.projection_vars
denominator_exponent(∇r::RoutingGradient) = ∇r.r.e

evaluate(∇r::RoutingGradient, x, p = nothing) = gradient(∇r.r, x, p)
evaluate!(u, ∇r::RoutingGradient, x, p = nothing) = gradient!(u, ∇r.r, x, p)
evaluate_and_jacobian(∇r::RoutingGradient, x, p = nothing) = gradient_and_hessian(∇r.r, x, p)
evaluate_and_jacobian!(u, U, ∇r::RoutingGradient, x, p = nothing) = gradient_and_hessian!(u, U, ∇r.r, x, p)

function taylor!(u, ::Val, F::RoutingGradient, x, p)
    fill!(u, zero(ComplexF64))
end