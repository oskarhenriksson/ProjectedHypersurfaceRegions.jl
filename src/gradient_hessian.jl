mutable struct GradientCache
    Ks::Vector{LinearSubspace}
    line_hypersurface_intersections::Vector
    tracker::EndgameTracker
    v::Vector
    H::Matrix
end

function GradientCache(k, PWS)
    n = ambient_dim(PWS)
    d = degree(PWS)

    Ks = Vector{LinearSubspace}(undef, k)
    line_hypersurface_intersections = [[zeros(ComplexF64, n) for _ = 1:d] for _ = 1:k]

    Hom = linear_subspace_homotopy(F, PWS.L, PWS.L; intrinsic = true)
    tracker = EndgameTracker(Hom)

    grad = zeros(ComplexF64, k)
    H = zeros(ComplexF64, k, k)

    GradientCache(Ks, line_hypersurface_intersections, tracker, grad, H)
end



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

    GC = GradientCache(k, PWS)

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
        hess = hess_log_r(PWS, k, e; c, B)
    elseif method == :many_slices
        hess = _many_slices(F_ordered, PWS, e, projection_vars; c, B)
    elseif method == :single_slice
        hess = _single_slice(F_ordered, PWS, e, projection_vars; c, B)
    end
    hess
end
#If you only have a pseudowitness set, then only one method remains.
function hess_log_r(
    PWS::PseudoWitnessSet,
    k::Int,
    e;
    c::Union{AbstractVector,Nothing} = nothing,
    B::Union{Matrix,Nothing} = nothing,
)

    if isnothing(c)
        c = randn(k)
    end
    if isnothing(B)
        B = Matrix(qr(randn(k, k)).Q)
    end
    GC = GradientCache(k, PWS)
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
                track_pws_to_lines!(GC, p, B[:, i] - B[:, j], PWS)
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
    F::System,
    PWS::PseudoWitnessSet,
    e,
    projection_vars::Vector{Variable};
    c::AbstractVector,
    B::Matrix,
)
    n = length(variables(F))
    k = length(projection_vars)
    @var u[1:n-k], t, p[1:k], b[1:k]
    JsuF = differentiate(F([p + (1 / t) * b; u]), vcat(t, u[:]))
    JpF = differentiate(F([p + (1 / t) * b; u]), p)
    q = 1 + sum((p - c) .* (p - c))
    Hlogqe(pt) = evaluate(differentiate(differentiate(log(q^e), p), p), p => pt) # Can expand to make faster?

    d = degree(PWS) # Could figure out a suitable e from d. Specifically, e >= d/2?

    GC = GradientCache(k, PWS)

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

function _single_slice(
    F::System,
    PWS::PseudoWitnessSet,
    e,
    projection_vars::Vector{Variable};
    c::Union{AbstractVector,Nothing} = nothing,
    B::Union{Matrix,Nothing} = nothing,
)
    # TODO: This needs to be implemented! For now, redirect to the off_diag method.
    hess_log_r(PWS, e, k; c, B)
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
