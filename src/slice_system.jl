struct RoutingGradient <: HC.AbstractSystem
    PWS::PseudoWitnessSet
    projection_vars::Vector{HC.Variable}
    GC::GradientCache
    e::Int
    c::Vector
    ∇logqe::Any
    ∇logprodg::Any
end
function RoutingGradient(
    F,
    projection_vars;
    e::Union{Int,Nothing} = nothing,
    c::Union{Vector,Nothing} = nothing,
    g::Union{Vector{Expression},Vector{Variable},Nothing} = nothing
)

    all_vars = variables(F)
    x_vars = setdiff(all_vars, projection_vars)
    F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
    k = length(projection_vars)
    PWS = PseudoWitnessSet(F_ordered, k)

    if isnothing(g) || length(g) == 0
        ∇logprodg = nothing
        g_degree = 0
    else
        g = Expression.(g)
        @assert variables(g) ⊆ projection_vars "Variables in g must match projection_vars"
        ∇logprodg = System(sum([differentiate(log(gi), projection_vars) for gi in g]), variables=projection_vars) |> fixed
        g_degree = sum(HC.degree.(g))
    end

    if isnothing(e)
        e = div(degree(PWS) + g_degree, 2) + 1
    end
    if isnothing(c)
        c = randn(k)
    end

    @var α[1:k]
    q = 1 + sum((α - c) .* (α - c))
    ∇logqe = System(differentiate(-e * log(q), α), variables = α) |> fixed

    # Use single-slice gradient cache to avoid tracking many lifted lines
    GC = GradientCache(PWS)

    RoutingGradient(PWS, projection_vars, GC, e, c, ∇logqe, ∇logprodg)
end

denominator_exponent(r::RoutingGradient) = r.e

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
    

    PWS, GC, ∇logqe, ∇logprodg = r.PWS, r.GC, r.∇logqe, r.∇logprodg

    evaluate!(u, ∇logqe, x)
    if !isnothing(∇logprodg)
        ∇logprodg_temp = GC.∇logprodg_temp
        evaluate!(∇logprodg_temp, ∇logprodg, x)
        u .+= ∇logprodg_temp
    end

    # Use cached symbolic objects and arrays
    JsuF = GC.JsuF
    JPF = GC.JPF
    JBF = GC.JBF

    v0 = GC.v0
    S = GC.S
    Uvals = GC.Uvals
    SB = GC.SB
    rhs1, rhs2, rhs3 = GC.rhs1, GC.rhs2, GC.rhs3

    N, n = size(PWS.F)
    k = n_projection_variables(PWS)

    # Track to PWS
    track!(GC, PWS, x)
    get_s_and_Uvals!(Uvals, S, GC, PWS)

    #Obtain gradients of S and U with respect to p and β
    for i = 1:length(S)

        if !PWS.track_report[i] # skip if i-th track failed
            continue
        end

        # fill v0 in-place instead of vcat
        v0[1] = S[i]
        @inbounds for ii = 1:size(Uvals,1)
            v0[1 + ii] = Uvals[ii, i]
        end
        @inbounds for ii = 1:length(x)
            v0[1 + size(Uvals,1) + ii] = x[ii]
        end

        # use indexed loops to avoid tuple allocations from enumerate
        JsuF_temp = GC.JsuF_temp
        for idx = 1:length(JsuF)
            evaluate!(rhs2, JsuF[idx], v0)
            @inbounds for ii in 1:N
                JsuF_temp[ii, idx] = rhs2[ii]
            end
        end

        JPF_temp = GC.JPF_temp
        for idx = 1:length(JPF)
            evaluate!(rhs2, JPF[idx], v0)
            @inbounds for ii in 1:N
                JPF_temp[ii, idx] = rhs2[ii]
            end
        end

        JBF_temp = GC.JBF_temp
        for idx = 1:length(JBF)
            evaluate!(rhs3, JBF[idx], v0)
            @inbounds for ii in 1:k
                JBF_temp[ii, idx] = rhs3[ii]
            end
        end

        # fill rhs1 in-place (unchanged) using explicit loops to avoid slice allocations
        for col = 1:size(JPF_temp, 2)
            @inbounds for row = 1:size(rhs1, 1)
                rhs1[row, col] = JPF_temp[row, col]
            end
        end
        for idx = 1:size(JBF_temp, 1)
            col = size(JPF_temp, 2) + idx
            @inbounds for row = 1:size(rhs1, 1)
                rhs1[row, col] = JBF_temp[idx, row]
            end
        end

        rhs1 .*= -1
        # In-place linear solving
        try
            Jsu0 = lu!(JsuF_temp)
            LinearAlgebra.ldiv!(Jsu0, rhs1) # solves the system Jsu*A = -[JP JB]
        catch
            fill!(rhs1, 0.0 + 0.0im)
        end


        # copy rhs1 row segment into SB row without creating slices
        for jj = 1:k
            SB[i, jj] = rhs1[1, k + jj]
        end
        @inbounds for jj = 1:k
            u[jj] -= SB[i, jj]
        end
    end


    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end


    nothing
end
function evaluate(r::RoutingGradient, x, p = nothing)

    m, n = size(r)
    u = zeros(ComplexF64, m)

    evaluate!(u, r, x, p)
    u
end
(r::RoutingGradient)(x) = evaluate(r, x)
function evaluate_and_jacobian!(u, U, r::RoutingGradient, x, p = nothing)

    PWS, GC, ∇logqe, ∇logprodg = r.PWS, r.GC, r.∇logqe, r.∇logprodg

    evaluate_and_jacobian!(u, U, ∇logqe, x)
    if !isnothing(∇logprodg)
        ∇logprodg_temp = GC.∇logprodg_temp
        Hess_logprodg_temp = GC.Hess_logprodg_temp
        evaluate_and_jacobian!(∇logprodg_temp, Hess_logprodg_temp, ∇logprodg, x)
        u .+= ∇logprodg_temp
        U .+= Hess_logprodg_temp
    end

    # Use cached symbolic objects and arrays
    JsuF = GC.JsuF
    JPF = GC.JPF
    JBF = GC.JBF
    HF = GC.HF
    JxB = GC.JxB
    JxP = GC.JxP
    JPB = GC.JPB

    # preallocate small temporaries to avoid allocating SubArray views inside inner loop
    temp_Hi = GC.temp_Hi
    temp_Jxpi = GC.temp_Jxpi
    temp_Jxbi = GC.temp_Jxbi
    temp_Jpbi = GC.temp_Jpbi

    v0 = GC.v0
    S = GC.S
    Uvals = GC.Uvals
    SP = GC.SP
    SB = GC.SB
    UP = GC.UP
    UB = GC.UB
    A = GC.A
    rhs1, rhs2, rhs3 = GC.rhs1, GC.rhs2, GC.rhs3

    M, M1, M2, M3 = GC.M, GC.M1, GC.M2, GC.M3

    k = n_projection_variables(PWS)
    N, n = size(PWS.F)

    # Track to PWS
    track!(GC, PWS, x)
    get_s_and_Uvals!(Uvals, S, GC, PWS)

    #Obtain gradients of S and U with respect to p and β
    for i = 1:length(S)

        if !PWS.track_report[i] # skip if i-th track failed
            continue
        end

        # fill v0 in-place instead of vcat
        v0[1] = S[i]
        @inbounds for ii = 1:size(Uvals,1)
            v0[1 + ii] = Uvals[ii, i]
        end
        @inbounds for ii = 1:length(x)
            v0[1 + size(Uvals,1) + ii] = x[ii]
        end

        # use indexed loops to avoid tuple allocations from enumerate
        JsuF_temp = GC.JsuF_temp
        for idx = 1:length(JsuF)
            evaluate!(rhs2, JsuF[idx], v0)
            @inbounds for ii in 1:N
                JsuF_temp[ii, idx] = rhs2[ii]
            end
        end

        JPF_temp = GC.JPF_temp
        for idx = 1:length(JPF)
            evaluate!(rhs2, JPF[idx], v0)
            @inbounds for ii in 1:N
                JPF_temp[ii, idx] = rhs2[ii]
            end
        end

        JBF_temp = GC.JBF_temp
        for idx = 1:length(JBF)
            evaluate!(rhs3, JBF[idx], v0)
            @inbounds for ii in 1:k
                JBF_temp[ii, idx] = rhs3[ii]
            end
        end

        # fill rhs1 in-place (unchanged) using explicit loops to avoid slice allocations
        for col = 1:size(JPF_temp, 2)
            @inbounds for row = 1:size(rhs1, 1)
                rhs1[row, col] = JPF_temp[row, col]
            end
        end
        for idx = 1:size(JBF_temp, 1)
            col = size(JPF_temp, 2) + idx
            @inbounds for row = 1:size(rhs1, 1)
                rhs1[row, col] = JBF_temp[idx, row]
            end
        end

        rhs1 .*= -1
        # In-place linear solving
        try
            Jsu0 = lu!(JsuF_temp)
            LinearAlgebra.ldiv!(Jsu0, rhs1) # solves the system Jsu*A = -[JP JB]
        catch
            fill!(rhs1, 0.0 + 0.0im)
        end


        # copy without slice allocations
        for jj= 1:k
            SP[i, jj] = rhs1[1, jj]
            SB[i, jj] = rhs1[1, k + jj]
        end
        for ii = 1:size(rhs1, 1)-1
            for jj = 1:k
                UP[i, ii, jj] = rhs1[1 + ii, jj]
                UB[i, ii, jj] = rhs1[1 + ii, k + jj]
            end
        end
        @inbounds for jj = 1:k
            u[jj] -= SB[i, jj]
        end

    end

    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end

    # Computation outlined in the abstract description Jon gave in Overleaf file
    for j = 1:length(S)

        !PWS.track_report[j] && continue # skip if j-th track failed

        # fill v0 in-place instead of vcat
        v0[1] = S[j]
        @inbounds for ii = 1:size(Uvals,1)
            v0[1 + ii] = Uvals[ii, j]
        end
        @inbounds for ii = 1:length(x)
            v0[1 + size(Uvals,1) + ii] = x[ii]
        end

        HF_temp = GC.HF_temp
        # indexed loops avoid allocations from eachcol/enumerate
        HF_nrows, HF_ncols = size(HF)
        for col_idx = 1:HF_ncols
            for row_idx = 1:HF_nrows
                evaluate!(rhs2, HF[row_idx, col_idx], v0)
                @inbounds for ii in 1:N
                    HF_temp[ii, row_idx, col_idx] = rhs2[ii]
                end
            end
        end

        # Evaluate JxB using evaluate! with indexed loops
        JxB_temp = GC.JxB_temp
        JxB_nrows, JxB_ncols = size(JxB)
        for col_idx = 1:JxB_ncols
            for row_idx = 1:JxB_nrows
                evaluate!(rhs2, JxB[row_idx, col_idx], v0)
                @inbounds for ii in 1:N
                    JxB_temp[ii, row_idx, col_idx] = rhs2[ii]
                end
            end
        end

        JxP_temp = GC.JxP_temp
        JxP_nrows, JxP_ncols = size(JxP)
        for col_idx = 1:JxP_ncols
            for row_idx = 1:JxP_nrows
                evaluate!(rhs2, JxP[row_idx, col_idx], v0)
                @inbounds for ii in 1:N
                    JxP_temp[ii, row_idx, col_idx] = rhs2[ii]
                end
            end
        end

        JPB_temp = GC.JPB_temp
        JPB_nrows, JPB_ncols = size(JPB)
        for col_idx = 1:JPB_ncols
            for row_idx = 1:JPB_nrows
                evaluate!(rhs2, JPB[row_idx, col_idx], v0)
                @inbounds for ii in 1:N
                    JPB_temp[ii, row_idx, col_idx] = rhs2[ii]
                end
            end
        end


        for i = 1:N

            # the following defines 
            #M1 = [SP[j, :] transpose(UP[j, :, :])] 
            #M2 = transpose([SB[j, :] transpose(UB[j, :, :])]) 
            for a = 1:k
                M1[a, 1] = SP[j, a]
            end
            for a = 1:k
                for b = 2:N
                    M1[a, b] = UP[j, b-1, a]
                end
            end
            for b = 1:k
                M2[1, b] = SB[j, b]
            end
            for b = 1:k
                for a = 2:N
                    M2[a, b] = UB[j, a-1, b]
                end
            end

            # copy slices into temporaries (avoids allocating SubArray objects)
            @inbounds for r = 1:HF_nrows, c = 1:HF_ncols
                temp_Hi[r, c] = HF_temp[i, r, c]
            end
            @inbounds for r = 1:JxP_nrows, c = 1:JxP_ncols
                temp_Jxpi[r, c] = JxP_temp[i, r, c]
            end
            @inbounds for r = 1:JxB_nrows, c = 1:JxB_ncols
                temp_Jxbi[r, c] = JxB_temp[i, r, c]
            end
            @inbounds for r = 1:JPB_nrows, c = 1:JPB_ncols
                temp_Jpbi[r, c] = JPB_temp[i, r, c]
            end

            # now step by step in-place matrix multiplications. 
            for a = 1:k, b = 1:k
                A[j, i, a, b] = temp_Jpbi[b, a] # note the transpose here
            end
            mul!(M, transpose(temp_Jxpi), M2)
            for a = 1:k, b = 1:k
                A[j, i, a, b] += M[b, a] # note the transpose here
            end
            mul!(M, M1, temp_Jxbi)
            for a = 1:k, b = 1:k
                A[j, i, a, b] += M[b, a] # note the transpose here
            end
            mul!(M3, M1, temp_Hi)
            mul!(M, M3, M2)
            for a = 1:k, b = 1:k
                A[j, i, a, b] += M[b, a] # note the transpose here
            end

        end
    end


    #Compute Hessian
    fill!(M, 0.0 + 0.0im) # here M will get assigned the Hessian of log r
    for j = 1:length(S)
        
        !PWS.track_report[j] && continue # skip if j-th track failed

        v0[1] = S[j]
        @inbounds for ii = 1:size(Uvals,1)
            v0[1 + ii] = Uvals[ii, j]
        end
        @inbounds for ii = 1:length(x)
            v0[1 + size(Uvals,1) + ii] = x[ii]
        end


        Jtu = GC.Jtu_temp
        for (idx, J) in enumerate(JsuF)
            evaluate!(rhs2, J, v0)
            @inbounds for ii in 1:N
                    Jtu[ii, idx] = rhs2[ii]
                end
        end
        Jtu0 = lu!(Jtu)
        for a = 1:k, b = 1:k
            for i = 1:N
                rhs2[i] = A[j, i, a, b]
            end
            LinearAlgebra.ldiv!(Jtu0, rhs2) # in-place linear algebra
            M[a, b] += rhs2[1]
        end
    end

    for a = 1:k, b = 1:k
        U[a, b] += M[a, b]
    end

    nothing
end
function evaluate_and_jacobian(r::RoutingGradient, x, p = nothing)

    m, n = size(r)
    u = zeros(ComplexF64, m)
    U = zeros(ComplexF64, m, n)
    evaluate_and_jacobian!(u, U, r, x, p)
    u, U
end

function taylor!(u, ::Val, F::RoutingGradient, x, p)
    for i = 1:length(u)
        u[i] = ComplexF64(0)
    end
end

## for testing
_evaluate(r::RoutingGradient, p::Vector) = evaluate(r, p)
_evaluate_evaluate_jacobian(r::RoutingGradient, p::Vector) = (r.H)(p)
