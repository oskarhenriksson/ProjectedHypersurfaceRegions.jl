



struct RoutingGradient <: HC.AbstractSystem
    F::HC.System
    projection_vars::Vector{HC.Variable}
    PWS::PseudoWitnessSet
    e::Int
    c::Vector
    B::Matrix
    ∇::Function
    H::Function
end
function RoutingGradient(F, projection_vars; 
                                e::Union{Int, Nothing} = nothing, 
                                B::Union{Matrix, Nothing} = nothing, 
                                c::Union{Vector, Nothing} = nothing)

    all_vars = variables(F)
    x_vars = setdiff(all_vars, projection_vars)
    F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
    k = length(projection_vars)
    PWS = PseudoWitnessSet(F_ordered, k, linear_subspace_codim = k - 1)
    
    if isnothing(e)
        e = floor(degree(PWS) / 2) + 1
    end
    if isnothing(c)
        c = randn(k)
    end
    if isnothing(B)
        B = Matrix(qr(randn(k, k)).Q)
    end

    ∇ = ∇log_r(PWS, k; e, c, B)
    H = hess_log_r(PWS, k, e; c, B)
    RoutingGradient(F_ordered, projection_vars, PWS, e, c, B, ∇, H)
end

(r::RoutingGradient)(x) = (r.∇)(x)

# Base.size(F::AbstractSystem)
# ModelKit.variables(F::AbstractSystem)::Vector{Variable}
# ModelKit.parameters(F::AbstractSystem) = Variable[]
# ModelKit.variable_groups(::AbstractSystem)::Union{Nothing,Vector{Vector{Variable}}} = nothing
#  # this has to work with x::Vector{Variable}
# (F::AbstractSystem)(x, p = nothing)
#  # this has to work with x::Vector{ComplexF64} and x::Vector{ComplexDF64}
# evaluate!(u, F::AbstractSystem, x, p = nothing)
# # this has only to work with x::Vector{ComplexF64}
# evaluate_and_jacobian!(u, U, F::AbstractSystem, x, p = nothing)
# If the system should be used in context of a parameter homotopy it is also necessary to implement

# taylor!(u, ::Val{1}, F::AbstractSystem, x, p::TaylorVector{2})
import Base.size
function Base.size(r::RoutingGradient) 
    k = length(r.projection_vars)
    (k, k)
end
ModelKit.variables(r::RoutingGradient) = r.projection_vars
#ModelKit.variables(∇::RoutingGradient) = ∇.params

import HomotopyContinuation.evaluate!
import HomotopyContinuation.evaluate_and_jacobian!
import HomotopyContinuation.evaluate
import HomotopyContinuation.evaluate_and_jacobian
function evaluate!(u, r::RoutingGradient, x, p = nothing)
    u .= (r.∇)(x)
    if !isnothing(p)
        u .= u - p
    end
    
    nothing
end
function evaluate(r::RoutingGradient, x, p = nothing)
    m, n = size(r)
    u = zeros(Float64, m)
    evaluate!(u, r, x, p)
    u  
end
function evaluate_and_jacobian!(u, U, r::RoutingGradient, x, p = nothing)
    u .= (r.∇)(x)
    U .= (r.H)(p)
    if !isnothing(p)
        u .= u - p
    end

    nothing
end
function evaluate_and_jacobian(r::RoutingGradient, x, p = nothing)
    m, n = size(r)
    u = zeros(Float64, m)
    U = zeros(Float64, m, n)
    evaluate_and_jacobian!(u, U, r, x, p)
    u, U    
end

# TODO: taylor!(u, ::Val{1}, F::AbstractSystem, x, p::TaylorVector{2})

## for testing
_evaluate(r::RoutingGradient, p::Vector) = (r.∇)(p)
_evaluate_jacobian(r::RoutingGradient, p::Vector) = (r.H)(p)


