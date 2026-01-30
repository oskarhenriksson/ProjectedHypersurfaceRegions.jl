#TODO: Work out some test case by hand (and add to the test file and README)

struct ProjectedHypersurface{TC} <: HC.AbstractSystem
    PWS::PseudoWitnessSet
    projection_vars::Vector{HC.Variable}
    GC::TC
end
function ProjectedHypersurface(
    F,
    projection_vars;
    PWS::Union{Nothing, PseudoWitnessSet} = nothing,
    start_system_for_PWS::Symbol = :polyhedral,
    compile::Union{Bool,Symbol} = :mixed
)

    all_vars = ModelKit.variables(F)
    x_vars = setdiff(all_vars, projection_vars)
    F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
    k = length(projection_vars)
    if isnothing(PWS)
        PWS = PseudoWitnessSet(F_ordered, k; start_system = start_system_for_PWS, compile = compile)
    end
    GC = GradientCache(PWS)

    ProjectedHypersurface{typeof(GC)}(PWS, projection_vars, GC)
end

degree(h::ProjectedHypersurface) = degree(h.PWS)

Base.show(io::IO, h::ProjectedHypersurface) = println(io, "Projected hypersurface of degree $(degree(h)) in ambient dimension $(nvariables(h))")

ModelKit.variables(h::ProjectedHypersurface{TC}) where {TC} = h.projection_vars
ModelKit.nvariables(h::ProjectedHypersurface{TC}) where {TC} = length(h.projection_vars)

function evaluate(h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}
    PWS, GC = h.PWS, h.GC

    S = GC.S
    X = GC.X
    Uvals = GC.Uvals

    k = n_projection_variables(PWS) #TODO: Remove?
    n = ambient_dim(PWS)

    track!(GC, PWS, x)

    u = 0.0

    get_s_and_Uvals!(Uvals, S, GC, PWS) # TODO: Is this what we want?    
    #@inbounds @simd 
    for si in S 
        u += -log(abs(si))
    end

    # TODO: How should we treat p?

    u
end

function gradient!(u, h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}
    
    PWS, GC = h.PWS, h.GC

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

    u .= zero(eltype(u))

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
        # In-place linear solving with pre-allocated pivot vector
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(JsuF_temp, GC.ipiv)
        if info == 0 # this indicates successful factorization
            LinearAlgebra.LAPACK.getrs!('N', JsuF_temp, ipiv, rhs1)
        else
            fill!(rhs1, zero(ComplexF64))
        end

        # copy rhs1 row segment into SB row without creating slices
        @inbounds @simd for jj = 1:k
            SB[i, jj] = rhs1[1, k + jj]
        end
        @inbounds @simd for jj = 1:k
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
function gradient(h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}
    k = nvariables(h)
    u = zeros(ComplexF64, k)
    gradient!(u, h, x, p)
    u
end

function gradient_and_hessian!(u, U, h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}

    PWS, GC = h.PWS, h.GC

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

    u .= zero(eltype(u))
    U .= zero(eltype(U))

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
        # In-place linear solving with pre-allocated pivot vector
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(JsuF_temp, GC.ipiv)
        if info == 0  # this indicates successful factorization
            LinearAlgebra.LAPACK.getrs!('N', JsuF_temp, ipiv, rhs1)
        else
            fill!(rhs1, zero(ComplexF64))
        end

        # copy without slice allocations
        @inbounds @simd for jj = 1:k
            SP[i, jj] = rhs1[1, jj]
            SB[i, jj] = rhs1[1, k + jj]
        end
        @inbounds for ii = 1:size(rhs1, 1)-1
            for jj = 1:k
                UP[i, ii, jj] = rhs1[1 + ii, jj]
                UB[i, ii, jj] = rhs1[1 + ii, k + jj]
            end
        end
        @inbounds @simd for jj = 1:k
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
    fill!(M, zero(ComplexF64)) # here M will get assigned the Hessian of log r
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
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(Jtu, GC.ipiv)
        for a = 1:k, b = 1:k
            for i = 1:N
                rhs2[i] = A[j, i, a, b]
            end
            if info == 0  # this indicates successful factorization
                LinearAlgebra.LAPACK.getrs!('N', Jtu, ipiv, rhs2)
            end
            M[a, b] += rhs2[1]
        end
    end

    for a = 1:k, b = 1:k
        U[a, b] += M[a, b]
    end

    nothing
end


function gradient_and_hessian(h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}

    k = nvariables(h)
    u = zeros(ComplexF64, k)
    U = zeros(ComplexF64, k, k)
    gradient_and_hessian!(u, U, h, x, p)
    u, U
end
