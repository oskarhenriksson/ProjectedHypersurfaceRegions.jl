



struct RoutingGradient <: HC.AbstractSystem
    F::HC.System
    projection_vars::Vector{HC.Variable}
    PWS::PseudoWitnessSet
    GC::GradientCache
    e::Int
    c::Vector
    B::Matrix
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

    # Use single-slice gradient cache to avoid tracking many lifted lines
    GC = GradientCache(PWS; single_slice = true)

    RoutingGradient(F_ordered, projection_vars, PWS, GC, e, c, B)
end

denominator_exponent(r::RoutingGradient) = r.e


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
import HomotopyContinuation.taylor!
function evaluate!(u, r::RoutingGradient, x, p = nothing)
    PWS, GC, e, c, B  = r.PWS, r.GC, r.e, r.c, r.B
    v = GC.v

    Qx = (sum((x - c) .^ 2) + 1, 2 .* (x - c))
    # For single-slice mode track only one lifted line and reuse its intersections
    if length(GC.Ks) == 1
        track_pws_to_line!(GC, x, B[:, 1], PWS)
        intersection_points = GC.line_hypersurface_intersections[1]
        for (i, bj) in enumerate(eachcol(B))
            v[i] = ∂log_r(intersection_points, Qx, e, x, bj)
        end
    else
        track_pws_to_lines!(GC, x, B, PWS)
        iter = zip(GC.line_hypersurface_intersections, eachcol(B))
        for (i, (intersection_points, bj)) in enumerate(iter)
            v[i] = ∂log_r(intersection_points, Qx, e, x, bj)
        end
    end

    u .= B * v # TODO: Cache this
    if !isnothing(p)
        u .= u - p
    end
    
    nothing
end
function evaluate(r::RoutingGradient, x, p = nothing)

    # T = eltype(x)
    # if T <: Real 
    #     T = Float64
    # end

    m, n = size(r)
    u = zeros(ComplexF64, m)

    evaluate!(u, r, x, p)
    u  
end
(r::RoutingGradient)(x) = evaluate(r, x)
function evaluate_and_jacobian!(u, U, r::RoutingGradient, x, p = nothing)

    PWS, GC, e, c, B  = r.PWS, r.GC, r.e, r.c, r.B
    v, H = GC.v, GC.H
    k = length(v)

    Qx = (sum((x - c) .^ 2) + 1, 2 .* (x - c))
    # Prefer single-line tracking to avoid many target_parameters! calls
    if length(GC.Ks) == 1
        track_pws_to_line!(GC, x, B[:, 1], PWS)
        intersection_points = GC.line_hypersurface_intersections[1]
        for (i, bj) in enumerate(eachcol(B))
            v[i] = ∂log_r(intersection_points, Qx, e, x, bj)
        end
    else
        track_pws_to_lines!(GC, x, B, PWS) #The only reason we track all the lines is so that we can evaluate.
        iter = zip(GC.line_hypersurface_intersections, eachcol(B))
        for (i, (intersection_points, bj)) in enumerate(iter)   
            v[i] = ∂log_r(intersection_points, Qx, e, x, bj)
        end
    end

    u .= B * v # TODO: Cache this
    if !isnothing(p)
        u .= u - p
    end

    ## Hessian
    ## TODO: Choose "best" Hessian and replace this calculation with that. Need new routing gradient to cache our stuff (that is computed once only).
    B = B[:,1]
    n = length(F.variables)
    k = n_projection_variables(PWS)
    @var u[1:n-k] p[1:k] β[1:k] t
    F_on_line = F([p + (1 / t) * β; u])
    N = length(F_on_line)
    # Symbolic Jacobians (except evaluated at β = B)
    JsuF = Expression.(evaluate(differentiate(F_on_line, vcat(t, u[:])), β => B)) #Jacobian of F with respect to s and u
    JPF = Expression.(evaluate(differentiate(F_on_line, p), β => B)) #Jacobian of F with respect to p
    JBF = Expression.(evaluate(differentiate(F_on_line, β), β => B)) #Jacobian of F with respect to \beta

    #Symbolic Hessians (except evaluated at β = B)
    HF = [Expression.(evaluate(differentiate(differentiate(F_on_line[i], vcat(t, u)), vcat(t, u)), β => B)) for i in 1:N]
    JxB = [Expression.(evaluate(differentiate(differentiate(F_on_line[i], vcat(t, u)), β), β => B)) for i in 1:N]
    JxP = [Expression.(evaluate(differentiate(differentiate(F_on_line[i], vcat(t, u)), p), β => B)) for i in 1:N]
    JPB = [Expression.(evaluate(differentiate(differentiate(F_on_line[i], p), β), β => B)) for i in 1:N]

    d= degree(PWS) 
    #Initializing several variables for use in the function
    # TODO: Consider not initializing these, or caching them for later.
    S = zeros(ComplexF64, d)
    Uvals = zeros(ComplexF64, n - k, d)

    SP = zeros(ComplexF64, length(S), k) 
    SB = zeros(ComplexF64, length(S), k)
    UP = zeros(ComplexF64, length(S), n - k, k)
    UB = zeros(ComplexF64, length(S), n - k, k)

    A = zeros(ComplexF64, length(S), size(F_on_line, 1), k, k)

    hess = zeros(ComplexF64, k, k)
    q = 1 + sum((p - c) .* (p - c))
    Hlogqe(pt) = evaluate(differentiate(differentiate(log(q^e), p), p), p => pt) 

    for (j, sol) in enumerate(GC.line_hypersurface_intersections[1])
        X = view(sol, 1:k)  # TODO: This creates new memory :( Workaround: Write for loop to copy values into X, which is cached. The main point is to avoid new allocations! This is why it is slow.
        # TODO: view creates a pointer rather than a copy. Might be bugged, try it. Same thing for U below.
    Uvals[:, j] = sol[k+1:end] # This creates a new vector in memory. should do a for loop.
        _ , nonzero_coordinate = findmax(abs, X - x)
        S[j] = B[nonzero_coordinate] / (X[nonzero_coordinate] - x[nonzero_coordinate]) # We solving for t inside of this: p + (1 / t) * β = X
    end

    #Obtain gradients of S and U with respect to p and β
    for i = 1:length(S)

    Jsu = evaluate(JsuF, vcat(t, u, p) => vcat(S[i], Uvals[:, i], x)) # TODO: Evaluate also creates memory... JsuF should be a vector of Systems, not expressions. This has evaluate!, which overwrites memory. Need to change this for ALL evaluates.
    JP = evaluate(JPF, vcat(t, u, p) => vcat(S[i], Uvals[:, i], x))
    JB = evaluate(JBF, vcat(t, u, p) => vcat(S[i], Uvals[:, i], x))

        PBsols = -Jsu \ [JP JB] # solves the system Jsu*A = -[JP JB]
        # TODO: This should be an "in-place" operation to avoid memory allocation.

        SP[i,:] = PBsols[1, 1:k]
        SB[i,:] = PBsols[1, k+1:end]

        UP[i,:,:] = PBsols[2:end, 1:k]
        UB[i,:,:] = PBsols[2:end, k+1:end]
    end

    # Computation outlined in the abstract description Jon gave in Overleaf file
    for i = 1:length(F_on_line)
        for j = 1:length(S)
            H = evaluate(HF[i], vcat(t, u, p) => vcat(S[j], Uvals[:, j], x))
            Jxb = evaluate(JxB[i], vcat(t, u, p) => vcat(S[j], Uvals[:, j], x))
            Jxp = evaluate(JxP[i], vcat(t, u, p) => vcat(S[j], Uvals[:, j], x))
            Jpb = evaluate(JPB[i], vcat(t, u, p) => vcat(S[j], Uvals[:, j], x))

            A[j, i, :, :] = ([SP[j, :] transpose(UP[j, :, :])] * H * transpose([SB[j, :] transpose(UB[j, :, :])])
                            + Jpb + [SP[j, :] transpose(UP[j, :, :])] * Jxb
                            + transpose(Jxp) * transpose([SB[j, :] transpose(UB[j, :, :])])) |> transpose |> Matrix

        end
    end
    
    #Compute Hessian
    for j = 1:length(S)
    Jtu = evaluate(JsuF, vcat(t, u, p) => vcat(S[j], Uvals[:, j], x)) 
        sols = zeros(ComplexF64, k, k)
        for a in 1:k, b in 1:k
            rhs = vcat([A[j, i, a, b] for i = 1:length(F_on_line)]...)
            sols[a, b] = -(Jtu\rhs)[1]
        end
        hess = hess - sols
    end


    U .= real(hess - Hlogqe(x))

    nothing
end
function evaluate_and_jacobian(r::RoutingGradient, x, p = nothing)
    
    # T = eltype(x)
    # if T <: Real 
    #     T = Float64
    # end

    m, n = size(r)
    u = zeros(ComplexF64, m)
    U = zeros(ComplexF64, m, n)
    evaluate_and_jacobian!(u, U, r, x, p)
    u, U    
end

function taylor!(u, ::Val, F::RoutingGradient, x, p)
    for i in 1:length(u)
        u[i] = ComplexF64(0)
    end
end

## for testing
_evaluate(r::RoutingGradient, p::Vector) = evaluate(r, p)
_evaluate_jacobian(r::RoutingGradient, p::Vector) = (r.H)(p)


