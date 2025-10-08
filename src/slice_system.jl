struct RoutingGradient <: HC.AbstractSystem
    PWS::PseudoWitnessSet
    projection_vars::Vector{HC.Variable}
    GC::GradientCache
    e::Int
    c::Vector
    B::Matrix
    ∇logqe
    Hlogqe
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

    @var α[1:k]
    q = 1 + sum((α - c) .* (α - c))
    ∇logqe(pt) = evaluate(differentiate(log(q^e), α), α => pt)
    Hlogqe(pt) = evaluate(differentiate(differentiate(log(q^e), α), α), α => pt) 

    # Use single-slice gradient cache to avoid tracking many lifted lines
    GC = GradientCache(PWS, B[:,1])

    RoutingGradient(PWS, projection_vars, GC, e, c, B, ∇logqe, Hlogqe)
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

import HomotopyContinuation.evaluate!
import HomotopyContinuation.evaluate_and_jacobian!
import HomotopyContinuation.evaluate
import HomotopyContinuation.taylor!

function evaluate!(u, r::RoutingGradient, x, p = nothing)
    PWS, GC, B, ∇logqe  = r.PWS, r.GC, r.B, r.∇logqe

    fill!(u, 0.0 + 0.0im)
    
    # Use cached symbolic objects and arrays
    JsuF = GC.JsuF
    JPF = GC.JPF
    JBF = GC.JBF

    S = GC.S
    X = GC.X
    Uvals = GC.Uvals
    SB = GC.SB
    rhs1 = GC.rhs1
    
    k = n_projection_variables(PWS)

    # TODO: perhaps we should always pass in a single column as B to routing gradient in the first place...
    track_pws_to_line!(GC, x, B[:,1], PWS)
  
    for (j, sol) in enumerate(GC.line_hypersurface_intersections[1])
        !GC.track_report[j] && continue # skip if j-th track failed
        @assert all(!isnan, sol) "NaN entries in intersection points: $sol"
        for idx in 1:k
            X[idx] = sol[idx]
        end
        # TODO: view creates a pointer rather than a copy. Might be bugged, try it. Same thing for U below.
        Uvals[:, j] = sol[k+1:end] # This creates a new vector in memory. should do a for loop.
        _ , nonzero_coordinate = findmax(abs, X - x)
        S[j] = B[nonzero_coordinate] / (X[nonzero_coordinate] - x[nonzero_coordinate]) # We solving for t inside of this: p + (1 / t) * β = X
    end

    #Obtain gradients of S and U with respect to p and β
    for i = 1:length(S)
        !GC.track_report[i] && continue # skip if i-th track failed

        v0 =  vcat(S[i], Uvals[:, i], x)

        JsuF_temp = GC.JsuF_temp
        for (idx, J) in enumerate(JsuF)
            evaluate!(view(JsuF_temp, :, idx), J, v0)
        end

        JPF_temp = GC.JPF_temp
        for (idx, J) in enumerate(JPF)
            evaluate!(view(JPF_temp, :, idx), J, v0)
        end
        
        JBF_temp = GC.JBF_temp
        for (idx, J) in enumerate(JBF)
            evaluate!(view(JBF_temp, :, idx), J, v0)
        end
        
        # Fill rhs in-place
        for col = 1:size(JPF_temp,2)
            rhs1[:, col] .= JPF_temp[:, col]
        end
        for idx = 1:size(JBF_temp,1)
            rhs1[:, size(JPF_temp,2)+idx] .= JBF_temp[idx, :]
        end

        rhs1 .*= -1
        # In-place linear solving
    Jsu0 = lu!(JsuF_temp) 
        LinearAlgebra.ldiv!(Jsu0, rhs1)
        
        SB[i,:] = rhs1[1, k+1:end]
        u .-= SB[i,:]
    end


    u .-= ∇logqe(x)
    if !isnothing(p)
        u .-= p
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

    PWS, GC, B, ∇logqe, Hlogqe = r.PWS, r.GC, r.B, r.∇logqe, r.Hlogqe

    fill!(u, 0.0 + 0.0im)

    # Use cached symbolic objects and arrays
    JsuF = GC.JsuF
    JPF = GC.JPF
    JBF = GC.JBF
    HF = GC.HF
    JxB = GC.JxB
    JxP = GC.JxP
    JPB = GC.JPB

    S = GC.S
    X = GC.X
    Uvals = GC.Uvals
    SP = GC.SP
    SB = GC.SB
    UP = GC.UP
    UB = GC.UB
    A = GC.A
    hess = GC.hess
    rhs1 = GC.rhs1
    rhs2 = GC.rhs2

    k = n_projection_variables(PWS)
    d = degree(PWS)
    N, n = size(PWS.F)

    track_pws_to_line!(GC, x, B[:,1], PWS)


    for (j, sol) in enumerate(GC.line_hypersurface_intersections[1])
        !GC.track_report[j] && continue # skip if j-th track failed
        for i in 1:k
            X[i] = sol[i]
        end
        for i in 1:(n-k)
            Uvals[i, j] = sol[k+i]
        end
        _ , nonzero_coordinate = findmax(abs, X - x)
        S[j] = B[nonzero_coordinate] / (X[nonzero_coordinate] - x[nonzero_coordinate]) # We solving for t inside of this: p + (1 / t) * β = X
    end

    #Obtain gradients of S and U with respect to p and β
    for i = 1:length(S)

        !GC.track_report[i] && continue # skip if i-th track failed

        v0 =  vcat(S[i], Uvals[:, i], x)
        @assert all(!isnan, v0) "NaN entries in v0: $v0"
       
        JsuF_temp = GC.JsuF_temp
        for (idx, J) in enumerate(JsuF)
            evaluate!(view(JsuF_temp, :, idx), J, v0)
        end
        
        JPF_temp = GC.JPF_temp
        for (idx, J) in enumerate(JPF)
            evaluate!(view(JPF_temp, :, idx), J, v0)
        end



        JBF_temp = GC.JBF_temp
        for (idx, J) in enumerate(JBF)
            evaluate!(view(JBF_temp, :, idx), J, v0)
        end
        
        # Fill rhs in-place
        for col = 1:size(JPF_temp,2)
            rhs1[:, col] .= JPF_temp[:, col]
        end
        for idx = 1:size(JBF_temp,1)
            rhs1[:, size(JPF_temp,2)+idx] .= JBF_temp[idx, :]
        end

        rhs1 .*= -1
        # In-place linear solving
    Jsu0 = lu!(JsuF_temp) 
        LinearAlgebra.ldiv!(Jsu0, rhs1) # solves the system Jsu*A = -[JP JB]

        SP[i,:] = rhs1[1, 1:k]
        SB[i,:] = rhs1[1, k+1:end]

        UP[i,:,:] = rhs1[2:end, 1:k]
        UB[i,:,:] = rhs1[2:end, k+1:end]

        u .-= SB[i,:]

    end


    u .-= ∇logqe(x)
    if !isnothing(p)
        u .-= p
    end

    # Computation outlined in the abstract description Jon gave in Overleaf file
    for j = 1:length(S)

        !GC.track_report[j] && continue # skip if j-th track failed

        v0 =  vcat(S[j], Uvals[:, j], x)

        HF_temp = GC.HF_temp
        for (col_idx, col) in enumerate(eachcol(HF))
            for (row_idx, h) in enumerate(col)
                evaluate!(view(HF_temp,:, row_idx,col_idx), h, v0)
            end
        end
        
        # Evaluate JxB using evaluate! instead of map with evaluate
        JxB_temp = GC.JxB_temp
        for (col_idx, col) in enumerate(eachcol(JxB))
            for (row_idx, h) in enumerate(col)
                evaluate!(view(JxB_temp,:, row_idx,col_idx), h, v0)
            end
        end

        JxP_temp = GC.JxP_temp
        for (col_idx, col) in enumerate(eachcol(JxP))
            for (row_idx, h) in enumerate(col)
                evaluate!(view(JxP_temp,:, row_idx,col_idx), h, v0)
            end
        end

        JPB_temp = GC.JPB_temp
        for (col_idx, col) in enumerate(eachcol(JPB))
            for (row_idx, h) in enumerate(col)
                evaluate!(view(JPB_temp,:, row_idx,col_idx), h, v0)
            end
        end
       

        for i = 1:N

            Hi = view(HF_temp, i, :, :)
            Jxpi = view(JxP_temp, i, :, :)
            Jxbi = view(JxB_temp, i, :, :) 
            Jpbi = view(JPB_temp, i, :, :)

            A[j, i, :, :] = ([SP[j, :] transpose(UP[j, :, :])] * Hi * transpose([SB[j, :] transpose(UB[j, :, :])])
                            + Jpbi + [SP[j, :] transpose(UP[j, :, :])] * Jxbi
                            + transpose(Jxpi) * transpose([SB[j, :] transpose(UB[j, :, :])])) |> transpose |> Matrix

        end
    end
    
    
    #Compute Hessian
    fill!(hess, 0.0 + 0.0im)
    for j = 1:length(S)
        !GC.track_report[j] && continue # skip if j-th track failed
        Jtu = GC.Jtu_temp
        for (idx, J) in enumerate(JsuF)
            evaluate!(view(Jtu, :, idx), J, vcat(S[j], Uvals[:, j], x))
        end
        Jtu0 = lu!(Jtu)
        for a in 1:k, b in 1:k
            for i in 1:N
                rhs2[i] = A[j, i, a, b]
            end
            LinearAlgebra.ldiv!(Jtu0, rhs2) # in-place linear algebra
            hess[a, b] += rhs2[1]
        end
    end

    U .= hess - Hlogqe(x)

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
_evaluate_evaluate_jacobian(r::RoutingGradient, p::Vector) = (r.H)(p)