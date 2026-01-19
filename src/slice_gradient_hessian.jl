mutable struct GradientCache{T}
    v0::Vector{T}
    line_hypersurface_intersections::Vector{Vector{T}}
    JsuF::Vector{HC.CompiledSystem}
    JPF::Vector{HC.CompiledSystem}
    JBF::Vector{HC.CompiledSystem}
    HF::Matrix{HC.CompiledSystem}
    JxB::Matrix{HC.CompiledSystem}
    JxP::Matrix{HC.CompiledSystem}
    JPB::Matrix{HC.CompiledSystem}
    S::Vector{T}
    X::Vector{T}
    Uvals::Matrix{T}
    SP::Matrix{T}
    SB::Matrix{T}
    UP::Array{T,3}
    UB::Array{T,3}
    A::Array{T,4}
    rhs1::Matrix{T}
    rhs2::Vector{T}
    rhs3::Vector{T}
    JsuF_temp::Matrix{T}
    JPF_temp::Matrix{T}
    JBF_temp::Matrix{T}
    Jtu_temp::Matrix{T} # Temporary storage for evaluating JsuF
    HF_temp::Array{T, 3} # Temporary storage for evaluating HF
    JxB_temp::Array{T, 3} # Temporary storage for evaluating JxB
    JxP_temp::Array{T, 3} # Temporary storage for evaluating JxP
    JPB_temp::Array{T, 3} # Temporary storage for evaluating JPB
    temp_Hi::Matrix{T}
    temp_Jxpi::Matrix{T}
    temp_Jxbi::Matrix{T}
    temp_Jpbi::Matrix{T}
    ipiv::Vector{LinearAlgebra.LAPACK.BlasInt} # allocation for pivot for lu! in place linear solving
    M::Matrix{T}
    M1::Matrix{T}
    M2::Matrix{T}
    M3::Matrix{T}
    ∇logprodg_temp::Vector{T}
    Hess_logprodg_temp::Matrix{T}
end
function compute_systems(F, n, k, B)
    @unique_var uval[1:n-k] α[1:k] β[1:k] t
    F_on_line = F([α + (1 / t) * β; uval])
    v = vcat(t, uval)
    vars = vcat(t, uval, α)

    ∇v = map(v) do vi
        HC.ModelKit.differentiate(F_on_line, vi) 
    end
    ∇α = map(α) do αi
        HC.ModelKit.differentiate(F_on_line, αi)
    end

    JsuF = map(∇v) do ∇vi
        g = evaluate(∇vi, β => B)
        System(g, variables = vars)
    end
    JPF = map(∇α) do ∇vi
        g = evaluate(∇vi, β => B)
        System(g, variables = vars)
    end
    JBF = map(F_on_line) do f
        g = evaluate(HC.ModelKit.differentiate(f, β), β => B)
        System(g, variables = vars)
    end

    function J(x) 
        map(Iterators.product(∇v, x)) do (∇vi, xj)
            hess_ij = evaluate(HC.ModelKit.differentiate(∇vi, xj), β => B)
            System(hess_ij, variables = vars) 
        end
    end

    HF = J(v)
    JxB = J(β)
    JxP = J(α)
    JPB = map(Iterators.product(∇α, β)) do (∇αi, βj)
            hess_ij = evaluate(HC.ModelKit.differentiate(∇αi, βj), β => B)
            System(hess_ij, variables = vars) 
        end
    return CompiledSystem.(JsuF), CompiledSystem.(JPF), CompiledSystem.(JBF), CompiledSystem.(HF), CompiledSystem.(JxB), CompiledSystem.(JxP), CompiledSystem.(JPB)

end

function GradientCache(PWS)
    d = degree(PWS)
    k = n_projection_variables(PWS)
    F = PWS.F
    L = PWS.L
    N, n = size(F)

    @assert N == n-k+1 "Unexpected length of system"

    line_hypersurface_intersections = [zeros(ComplexF64, n + 1) for _ in 1:d]
  
    @unique_var t, p[1:k]

    S = zeros(ComplexF64, d)
    X = zeros(ComplexF64, k)
    Uvals = zeros(ComplexF64, n - k, d)
    SP = zeros(ComplexF64, d, k)
    SB = zeros(ComplexF64, d, k)
    UP = zeros(ComplexF64, d, n - k, k)
    UB = zeros(ComplexF64, d, n - k, k)
    A = zeros(ComplexF64, d, N, k, k) 

    JsuF, JPF, JBF, HF, JxB, JxP, JPB = compute_systems(F, n, k, L.b)


    # 
    rhs1 = zeros(ComplexF64, N, 2*k)  
    rhs2 = zeros(ComplexF64, N)  
    rhs3 = zeros(ComplexF64, k)  

    JsuF_temp = zeros(ComplexF64, N, 1+n-k)
    JPF_temp = zeros(ComplexF64, N, k)
    JBF_temp = zeros(ComplexF64, k, N)
    Jtu_temp = zeros(ComplexF64, N, 1+n-k) # TODO: Maybe can reuse Jsu_temp....
    HF_temp = zeros(ComplexF64, N, size(HF)...)
    JxB_temp = zeros(ComplexF64, N, size(JxB)...) # size(JxB)
    JxP_temp = zeros(ComplexF64, N, size(JxP)...)
    JPB_temp = zeros(ComplexF64, N, size(JPB)...)
    temp_Hi = zeros(ComplexF64, size(HF)...)
    temp_Jxpi = zeros(ComplexF64, size(JxB)...)
    temp_Jxbi = zeros(ComplexF64, size(JxP)...)
    temp_Jpbi = zeros(ComplexF64, size(JPB)...)

    ipiv = Vector{LinearAlgebra.LAPACK.BlasInt}(undef, min(size(JsuF_temp,1), size(JsuF_temp,2)))

    M = zeros(ComplexF64, k, k)
    M1 = zeros(ComplexF64, k, n-k+1)
    M2 = zeros(ComplexF64, n-k+1, k)
    M3 = zeros(ComplexF64, k, n-k+1)

    ∇logprodg_temp = zeros(ComplexF64, k)
    Hess_logprodg_temp = zeros(ComplexF64, k, k)

    v0 = randn(ComplexF64, n+1)

    GradientCache{ComplexF64}(v0, 
                    line_hypersurface_intersections,
                    JsuF,
                    JPF,
                    JBF,
                    HF,
                    JxB,
                    JxP,
                    JPB,
                    S, 
                    X, 
                    Uvals, 
                    SP, 
                    SB, 
                    UP, 
                    UB, 
                    A, 
                    rhs1, 
                    rhs2, 
                    rhs3,
                    JsuF_temp, 
                    JPF_temp, 
                    JBF_temp, 
                    Jtu_temp, 
                    HF_temp, 
                    JxB_temp, 
                    JxP_temp, 
                    JPB_temp, 
                    temp_Hi,
                    temp_Jxpi,
                    temp_Jxbi,
                    temp_Jpbi,
                    ipiv,
                    M, 
                    M1, 
                    M2, 
                    M3, 
                    ∇logprodg_temp, 
                    Hess_logprodg_temp
                )
end

track!(GC::GradientCache, PWS::PseudoWitnessSet, p) = track!(GC.line_hypersurface_intersections, PWS, p) 

∇log_r(F::Vector{Expression}, k::Int; kwargs...) =
    ∇log_r(System(F), variables(F)[1:k]; kwargs...)
∇log_r(F::Vector{Expression}, projection_vars::Vector{Variable}; kwargs...) =
    ∇log_r(System(F), projection_vars; kwargs...)
∇log_r(F::System, k::Int; kwargs...) = ∇log_r(F, variables(F)[1:k]; kwargs...)
function ∇log_r(F::System, projection_vars::Vector{Variable}; kwargs...)

    all_vars = variables(F)
    x_vars = setdiff(all_vars, projection_vars)
    F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
    k = length(projection_vars)
    PWS = PseudoWitnessSet(F_ordered, k, linear_subspace_codim = k - 1)

    ∇log_r(PWS, k; kwargs...)
end


function ∇log_r(
    PWS::PseudoWitnessSet,
    k::Int;
    e::Union{Number, Nothing} = nothing,
    c::Union{AbstractVector,Nothing} = nothing,
    B::Union{Matrix,Nothing} = nothing,
)
    if isnothing(e)
        e = floor(degree(PWS) / 2) + 1
    end
    if isnothing(c)
        c = randn(k)
    end
    if isnothing(B)
        B = Matrix(qr(randn(k, k)).Q)
    end

    GC = GradientCache(PWS)

    Q(p) = (sum((p - c) .^ 2) + 1, 2 .* (p - c))

    function f(p)
        Qp = Q(p)

        track_pws_to_lines!(GC, p, B, PWS)

        # out = map(
        #     zip(GC.line_hypersurface_intersections, eachcol(B)),
        # ) do (intersection_points, bj)
        #     ∂log_r(intersection_points, Qp, e, p, bj)
        # end
        iter = zip(GC.line_hypersurface_intersections, eachcol(B))
        for (i, (intersection_points, bj)) in enumerate(iter)   
            GC.v[i] = ∂log_r(intersection_points, Qp, e, p, bj)
        end

        real(B * GC.v)
    end

    return f
end




### Computing Derivatives ###
function ∂log_h(
    intersection_points::AbstractVector{<:AbstractVector{<:Number}},
    p::AbstractVector,
    bj::AbstractVector,
)
    k = length(bj)
    i = findfirst(x -> abs(x) > 1e-5, bj)
    projection_points = map(s -> s[1:k], intersection_points)
    s = map(y -> (y[i] - p[i]) / bj[i], projection_points)

    -sum(1 / si for si in s)
end

function ∂log_qe(qp, ∇qp::AbstractVector, e, bj::AbstractVector)
    -e * (transpose(∇qp) * bj) / qp
end

function ∂log_r(
    intersection_points::AbstractVector{<:AbstractVector{<:Number}},
    Qp::Tuple,
    e,
    p::AbstractVector,
    bj::AbstractVector,
)
    qp, ∇qp = Qp
    ∂log_h(intersection_points, p, bj) + ∂log_qe(qp, ∇qp, e, bj)
end




### Computing Second Derivatives ###

function ∂2log_h(
    intersection_points::AbstractVector{<:AbstractVector{<:Number}},
    p::AbstractVector,
    bj::AbstractVector,
)
    k = length(bj)
    i = findfirst(x -> abs(x) > 1e-5, bj)
    projection_points = map(s -> s[1:k], intersection_points)
    s = map(y -> (y[i] - p[i]) / bj[i], projection_points)

    -sum(1 / si^2 for si in s)
end

function ∂2log_qe(Qp::Tuple, e, bj::AbstractVector)
    (qp, ∇qp) = Qp
    -e * (transpose(bj) * 2 * bj * qp - (transpose(∇qp) * bj)^2) / qp^2
end

function ∂2log_r(
    intersection_points::AbstractVector{<:AbstractVector{<:Number}},
    Qp::Tuple,
    e,
    p::AbstractVector,
    bj::AbstractVector,
)
    ∂2log_h(intersection_points, p, bj) + ∂2log_qe(Qp, e, bj)
end

function compute_off_diag(intermediate, bi_val, bj_val)
    (intermediate - bi_val - bj_val) / 2
end

# The following allows you to be lazy and not input a System.
hess_log_r(F::Vector{Expression}, e, projection_vars::Vector{Variable}; kwargs...) =
    hess_log_r(System(F), e, projection_vars; kwargs...)
hess_log_r(F::System, e, k::Int; kwargs...) =
    hess_log_r(F, e, variables(F)[1:k]; kwargs...)
# The following function will determine the method of computing the hessian you want and send it off.
function hess_log_r(
    F::System,
    e,
    projection_vars::Vector{Variable};
    method::Symbol = :off_diag,
    c::Union{AbstractVector,Nothing} = nothing,
    B::Union{Matrix,Nothing} = nothing,
)
    k = length(projection_vars)
    if isnothing(c)
        c = randn(k) # Is there any use in having a random c? probably not.
    end
    if isnothing(B)
        B = Matrix(qr(randn(k, k)).Q)
    end
    all_vars = variables(F)
    x_vars = setdiff(all_vars, projection_vars)
    F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
    PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1)

    if method == :off_diag
        hess = _off_diag(PWS, e, c, B) 
    elseif method == :many_slices
        hess = _many_slices(PWS, e, c, B)
    elseif method == :single_slice
        hess = _single_slice(PWS, e, c, B[:,1])
    end
    hess
end

function _off_diag(
    PWS::PseudoWitnessSet,
    e,
    c::AbstractVector,
    B::Matrix
)

    k = n_projection_variables(PWS)

    GC = GradientCache(PWS)
    Q(p) = (sum((p - c) .^ 2) + 1, 2 .* (p - c))

    function f(p)
        Qp = Q(p)

        track_pws_to_lines!(GC, p, B, PWS)

        iter = zip(GC.line_hypersurface_intersections, eachcol(B))
        for (i, (intersection_points, bj)) in enumerate(iter)   
            GC.v[i] = ∂2log_r(intersection_points, Qp, e, p, bj)
        end
        # diagonals = map(
        #     zip(GC.line_hypersurface_intersections, eachcol(B)),
        # ) do (intersection_points, bj)
        #     ∂2log_r(intersection_points, Qx, e, x, bj)
        # end
        for i in 1:k
            GC.H[i,i] = GC.v[i]
        end
        for i = 1:k
            for j = i+1:k
                track_pws_to_line!(GC, p, B[:, i] - B[:, j], PWS)
                intermediate = ∂2log_r(
                    GC.line_hypersurface_intersections[1],
                    Qp,
                    e,
                    p,
                    B[:, i] - B[:, j],
                )
                hij = -compute_off_diag(intermediate, GC.v[i], GC.v[j])
                GC.H[i, j] = hij
                GC.H[j, i] = hij
            end
        end
        real(B * GC.H * B^(-1)) #Return the hessian in the standard basis.
        # diagonals = map(
        #     zip(GC.line_hypersurface_intersections, eachcol(B)),
        # ) do (intersection_points, bj)
        #     ∂2log_r(intersection_points, Qp, e, p, bj)
        # end
        # H = diagm(diagonals)
        # for i = 1:k
        #     for j = i+1:k
        #         track_pws_to_lines!(GC, p, B[:, i] - B[:, j], PWS)
        #         intermediate = ∂2log_r(
        #             GC.line_hypersurface_intersections[1],
        #             Qp,
        #             e,
        #             p,
        #             B[:, i] - B[:, j],
        #         )
        #         H[i, j] = -compute_off_diag(intermediate, diagonals[i], diagonals[j])
        #         H[j, i] = -compute_off_diag(intermediate, diagonals[i], diagonals[j])
        #     end
        # end
        # real(B * H * B^(-1)) #Return the hessian in the standard basis.
    end
    f
end


#Jon mentioned reparametrizing our line from p + t b to tp - b so that our sum goes from -∑1/s_i to (do i add a negative here?)∑s_i.
#That is, I replace l = p + tb ---> t^{-1} l = t^{-1} p + b  ---> l' = t*p+b. This is now
#implemented and seems to perform more accurately than the original parametrization.
function _many_slices(
    PWS::PseudoWitnessSet,
    e,
    c::AbstractVector,
    B::Matrix,
)
    F = system(PWS)
    n = ambient_dim(PWS)
    k = n_projection_variables(PWS)
    @var u[1:n-k], t, p[1:k], b[1:k]
    JsuF = differentiate(F([p + (1 / t) * b; u]), vcat(t, u[:]))
    JpF = differentiate(F([p + (1 / t) * b; u]), p)
    q = 1 + sum((p - c) .* (p - c))
    Hlogqe(pt) = evaluate(differentiate(differentiate(log(q^e), p), p), p => pt) # Can expand to make faster?

    d = degree(PWS)
    
    GC = GradientCache(PWS)

    function f(P)
        track_pws_to_lines!(GC, P, B, PWS) # Our u's 

        H = zeros(ComplexF64, k, k)
        for bcol = 1:k
            num_intersections = length(GC.line_hypersurface_intersections[bcol])
            S = zeros(ComplexF64, num_intersections)
            ∇pS = zeros(ComplexF64, k, num_intersections)
            for i = 1:num_intersections
                upoint = GC.line_hypersurface_intersections[bcol][i] # world coordinates of the intersection point prior to projection
                j = argmax(abs.(B[:, bcol]))
                S[i] = B[j, bcol] / (upoint[j] - P[j]) # The value of t that corresponds to the intersection point 
                subs_dict = Dict(
                    [p[j] => P[j] for j in eachindex(p)]...,
                    [b[j] => B[j, bcol] for j in eachindex(b)]...,
                    t => S[i],
                    [u[j] => upoint[k+j] for j in eachindex(u)]...,
                )
                JsuF_eval = evaluate(JsuF, subs_dict)
                JpF_eval = evaluate(JpF, subs_dict)
                sols = JsuF_eval \ (-JpF_eval) # This is a matrix [∇p s; Jp u]
                ∇pS[:, i] = sols[1, :]
            end
            # Now, we compute the sum H(log(h(p))) * b = ∑_j^d 1/s_j^{-2} (∇p s_j)
            # h|L(t) = k(t-s_1)(t-s_2)
            # ∇h/h*b = -1/s_1 - 1/s_2
            # but, our S[1] = 1/s_1 
            # = -S[1] - S[2]
            # So if you take the derivative of this
            # = -∇S[1] - ∇S[2]
            Hlogh = -1 * sum(eachcol(∇pS))

            H[:, bcol] = Hlogh - Hlogqe(P) * B[:, bcol]
        end
        real(H * B^(-1)) # How to not to inverse?
    end
    f
    # TODO: I am overcomputing here since the Hessian is symmetric. Also, to get each ∇s_p I think I'm solving a larger system than needed. Perhaps there is a way to only compute the upper triangular part?

end


# TODO: Fix function below to compute hessian AND the gradient (at the same time)
# Most importantly, both should reuse the output of track_pws_to_line!
# Some things don't need to be computed everytime, and some things we do. We should isolate those.
# Cache everything below into GC.
# TODO: Put f(P) in evaluate_and_jacobian!
function _single_slice(
    PWS::PseudoWitnessSet,
    e,
    c::AbstractVector,
    B::Vector
)

    F = PWS.F
    
    n = length(F.variables)
    k = n_projection_variables(PWS)

    @var u[1:n-k] p[1:k] β[1:k] t
    F_on_line = F([p + (1 / t) * β; u])
    N = length(F_on_line)
    @assert N == n-k+1 "Unexpected length of system"

    #since B remains the same no matter what P, it makes sense to evaluate
    #the symbolic expressions at B here instead of when the function f(P) is called
    # Symbolic Jacobians (except evaluated at β = B)
    JsuF = Expression.(evaluate(differentiate(F_on_line, vcat(t, u[:])), β => B)) #Jacobian of F with respect to s and u
    JPF = Expression.(evaluate(differentiate(F_on_line, p), β => B)) #Jacobian of F with respect to p
    JBF = Expression.(evaluate(differentiate(F_on_line, β), β => B)) #Jacobian of F with respect to \beta

    #Symbolic Hessians (except evaluated at β = B)
    HF = [Expression.(evaluate(differentiate(differentiate(F_on_line[i], vcat(t, u)), vcat(t, u)), β => B)) for i in 1:N]
    JxB = [Expression.(evaluate(differentiate(differentiate(F_on_line[i], vcat(t, u)), β), β => B)) for i in 1:N]
    JxP = [Expression.(evaluate(differentiate(differentiate(F_on_line[i], vcat(t, u)), p), β => B)) for i in 1:N]
    JPB = [Expression.(evaluate(differentiate(differentiate(F_on_line[i], p), β), β => B)) for i in 1:N]

    GC = GradientCache(PWS; single_slice = true)
    
    d= degree(PWS) 
    #Initializing several variables for use in the function
    # TODO: Consider not initializing these, or caching them for later.
    S = zeros(ComplexF64, d)
    U = zeros(ComplexF64, n - k, d)

    SP = zeros(ComplexF64, length(S), k) 
    SB = zeros(ComplexF64, length(S), k)
    UP = zeros(ComplexF64, length(S), n - k, k)
    UB = zeros(ComplexF64, length(S), n - k, k)

    A = zeros(ComplexF64, length(S), size(F_on_line, 1), k, k)

    hess = zeros(ComplexF64, k, k)

    # Set up function for hess(q)
    q = 1 + sum((p - c) .* (p - c))
    Hlogqe(pt) = evaluate(differentiate(differentiate(log(q^e), p), p), p => pt) 

    function f(P) 

        fill!(hess, 0)

        # Compute the intersection points through a pseudowitness set
        track_pws_to_line!(GC, P, B, PWS) 
        #Obtain U and the projection S

        for (j, sol) in enumerate(GC.line_hypersurface_intersections[1])
            X = view(sol, 1:k)  # TODO: This creates new memory :( Workaround: Write for loop to copy values into X, which is cached. The main point is to avoid new allocations! This is why it is slow.
            # TODO: view creates a pointer rather than a copy. Might be bugged, try it. Same thing for U below.
            U[:, j] = sol[k+1:end] # This creates a new vector in memory. should do a for loop.
            _, nonzero_coordinate = findmax(abs, X - P)
            S[j] = B[nonzero_coordinate] / (X[nonzero_coordinate] - P[nonzero_coordinate]) # We solving for t inside of this: p + (1 / t) * β = X
        end

        #Obtain gradients of S and U with respect to p and β
        for i = 1:length(S)

            Jsu = evaluate(JsuF, vcat(t, u, p) => vcat(S[i], U[:, i], P)) # TODO: Evaluate also creates memory... JsuF should be a vector of Systems, not expressions. This has evaluate!, which overwrites memory. Need to change this for ALL evaluates.
            JP = evaluate(JPF, vcat(t, u, p) => vcat(S[i], U[:, i], P))
            JB = evaluate(JBF, vcat(t, u, p) => vcat(S[i], U[:, i], P))

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
                H = evaluate(HF[i], vcat(t, u, p) => vcat(S[j], U[:, j], P))
                Jxb = evaluate(JxB[i], vcat(t, u, p) => vcat(S[j], U[:, j], P))
                Jxp = evaluate(JxP[i], vcat(t, u, p) => vcat(S[j], U[:, j], P))
                Jpb = evaluate(JPB[i], vcat(t, u, p) => vcat(S[j], U[:, j], P))

                A[j, i, :, :] = ([SP[j, :] transpose(UP[j, :, :])] * H * transpose([SB[j, :] transpose(UB[j, :, :])])
                                + Jpb + [SP[j, :] transpose(UP[j, :, :])] * Jxb
                                + transpose(Jxp) * transpose([SB[j, :] transpose(UB[j, :, :])])) |> transpose |> Matrix

            end
        end
        
        #Compute Hessian
        for j = 1:length(S)
            Jtu = evaluate(JsuF, vcat(t, u, p) => vcat(S[j], U[:, j], P)) 
            sols = zeros(ComplexF64, k, k)
            for a in 1:k, b in 1:k
                rhs = vcat([A[j, i, a, b] for i = 1:length(F_on_line)]...)
                sols[a, b] = -(Jtu\rhs)[1]
            end
            hess = hess - sols
        end

        GC.H = hess - Hlogqe(P) 
        real(GC.H)
    end
    f
end

# If you happen to know the discriminant directly (and are not taking a projection), then this function can be used. 
# This is useful for testing our other hessian methods.
function hess_log_r_given_h(
    h::Expression,
    e;
    c::Union{AbstractVector,Nothing} = nothing,
)
    if isnothing(c)
        c = randn(length(variables(h)))
    end
    p = variables(h)
    q = 1 + sum((p - c) .* (p - c))
    r = log(h / q^e)
    H = differentiate(differentiate(r, p), p)
    hess(P) = evaluate(H, p => P)
    hess
end
