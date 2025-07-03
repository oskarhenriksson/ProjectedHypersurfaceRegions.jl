



struct RoutingGradient <: HC.AbstractSystem
    F::HC.System
    projection_vars::Vector{HC.Variable}
    PWS::PseudoWitnessSet
    GC::GradientCache
    e::Int
    c::Vector
    B::Matrix
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

    GC = GradientCache(k, PWS)

    H = hess_log_r(PWS, k, e; c, B)
    RoutingGradient(F_ordered, projection_vars, PWS, GC, e, c, B, H)
end



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
function evaluate!(u, r::RoutingGradient, x, p = nothing)
    PWS, GC, e, c, B  = r.PWS, r.GC, r.e, r.c, r.B
    v = GC.v

    Qx = (sum((x - c) .^ 2) + 1, 2 .* (x - c))
    track_pws_to_lines!(GC, x, B, PWS)

    # out = map( # TODO: Cache this
    #     zip(GC.line_hypersurface_intersections, eachcol(B)),
    # ) do (intersection_points, bj)
    #     ∂log_r(intersection_points, Qx, e, x, bj)
    # end
    iter = zip(GC.line_hypersurface_intersections, eachcol(B))
    for (i, (intersection_points, bj)) in enumerate(iter)   
        v[i] = ∂log_r(intersection_points, Qx, e, x, bj)
    end

    u .= real(B * v) # TODO: Cache this
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
(r::RoutingGradient)(x) = evaluate(r, x)
function evaluate_and_jacobian!(u, U, r::RoutingGradient, x, p = nothing)

    PWS, GC, e, c, B  = r.PWS, r.GC, r.e, r.c, r.B
    v, H = GC.v, GC.H

    Qx = (sum((x - c) .^ 2) + 1, 2 .* (x - c))
    track_pws_to_lines!(GC, x, B, PWS)

    ## Gradient
    # out = map( # TODO: Cache this
    #     zip(GC.line_hypersurface_intersections, eachcol(B)),
    # ) do (intersection_points, bj)
    #     ∂log_r(intersection_points, Qx, e, x, bj)
    # end
    iter = zip(GC.line_hypersurface_intersections, eachcol(B))
    for (i, (intersection_points, bj)) in enumerate(iter)   
        v[i] = ∂log_r(intersection_points, Qx, e, x, bj)
    end

    u .= real(B * v) # TODO: Cache this
    if !isnothing(p)
        u .= u - p
    end

    ## Hessian
    for (i, (intersection_points, bj)) in enumerate(iter)   
        v[i] = ∂2log_r(intersection_points, Qx, e, x, bj)
    end
    # diagonals = map(
    #     zip(GC.line_hypersurface_intersections, eachcol(B)),
    # ) do (intersection_points, bj)
    #     ∂2log_r(intersection_points, Qx, e, x, bj)
    # end
    for i in 1:k
        H[i,i] = v[i]
    end
    for i = 1:k
        for j = i+1:k
            track_pws_to_lines!(GC, x, B[:, i] - B[:, j], PWS)
            intermediate = ∂2log_r(
                GC.line_hypersurface_intersections[1],
                Qx,
                e,
                x,
                B[:, i] - B[:, j],
            )
            hij = -compute_off_diag(intermediate, v[i], v[j])
            H[i, j] = hij
            H[j, i] = hij
        end
    end
    U .= real(B * H * B^(-1)) #Return the hessian in the standard basis.

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
_evaluate(r::RoutingGradient, p::Vector) = evaluate(r, p)
_evaluate_jacobian(r::RoutingGradient, p::Vector) = (r.H)(p)


