  

struct PseudoWitnessSet
    F::System
    L::LinearSubspace
    W::Result
end
degree(PWS::PseudoWitnessSet) = length(PWS.W)

function PseudoWitnessSet(F::System, L::LinearSubspace) 
    n = ambient_dim(L)
    startL = rand_subspace(n; codim = 1)
    S = solutions(witness_set(F, startL))
    W = HC.solve(F, S, start_subspace = startL, target_subspace = L, intrinsic = true)
    PseudoWitnessSet(F, L, W)
end


∇log_r(F::Vector{Expression}, k::Int; kwargs...) = ∇log_r(System(F), variables(F)[1:k]; kwargs ...)
∇log_r(F::Vector{Expression}, projection_vars::Vector{Variable}; kwargs...) = ∇log_r(System(F), projection_vars; kwargs ...)
∇log_r(F::System, k::Int; kwargs...) = ∇log_r(F, variables(F)[1:k]; kwargs ...)
function ∇log_r(
    F::System,
    projection_vars::Vector{Variable};
    kwargs...
) 
    
    all_vars = variables(F)
    x_vars = setdiff(all_vars, projection_vars)
    F0 = System(F.expressions, variables = [projection_vars; x_vars])
    k = length(projection_vars)
    u = rand(ComplexF64, k)
    v = rand(ComplexF64, k)
    PWS = PseudoWitnessSet(F0, create_line(u, v, size(F0, 2)))

    ∇log_r(PWS, k; kwargs...)
end


function ∇log_r(
    PWS::PseudoWitnessSet,
    k::Int;
    e::Union{Real, Nothing} = nothing,
    c::Union{AbstractVector{<:Real}, Nothing} = nothing,
    B::Union{Matrix{<:Real}, Nothing} = nothing
)
    if isnothing(e)
        e = floor(degree(PWS)/2) + 1
    end
    if isnothing(c)
        c = randn(k)
    end
    if isnothing(B)
        B = Matrix(qr(randn(k,k)).Q)
    end
    
    Q(p) = (sum((p - c) .^ 2) + 1, 2 .* (p - c))

    function f(p)
        Qp = Q(p)

        line_hypersurface_intersections = track_pws_to_lines(p, B, PWS)

        out = map(zip(line_hypersurface_intersections, eachcol(B))) do (intersection_points, bj)
            ∂log_r(intersection_points, Qp, e, p, bj)
        end

        real(B* out)
    end

    return f
end





"""Returns a LinearSubspace of the form u + tv in R^n"""
function create_line(u::AbstractVector{<:Number}, v::AbstractVector{<:Number}, n::Int64)
    k = length(u)
    πA = u' - (dot(u, v)/dot(v, v)) * v'
    A = [πA zeros(n-k)]
    b = [πA * u]

    LinearSubspace(A, b)
end


function track_pws_to_lines(
    p::AbstractVector{<:Real},
    B::AbstractArray{Float64},
    PWS::PseudoWitnessSet,
)
    L = PWS.L
    Ks = map(bj -> create_line(p, bj, ambient_dim(L)), eachcol(B))
    HC.solve(
        PWS.F,
        PWS.W,
        start_subspace = L,
        target_subspaces = Ks,
        intrinsic = true,
        transform_result = (r, p) -> solutions(r),
    )
end



### Computing Derivatives ###
function ∂log_h(
    intersection_points::AbstractVector{<:AbstractVector{<:Number}},
    p::AbstractVector{<:Real},
    bj::AbstractVector{<:Real},
)
    k = length(bj)
    i = findfirst(x->abs(x) > 1e-5, bj)
    projection_points = map(s -> s[1:k], intersection_points)
    s = map(y -> (y[i] - p[i])/bj[i], projection_points)

    -sum(1/si for si in s)
end

function ∂log_qe(qp::Real, ∇qp::AbstractVector{<:Real}, e::Real, bj::AbstractVector{<:Real})
    -e * (transpose(∇qp) * bj)/qp
end

function ∂log_r(
    intersection_points::AbstractVector{<:AbstractVector{<:Number}},
    Qp::Tuple,
    e::Real,
    p::AbstractVector{<:Real},
    bj::AbstractVector{<:Real},
)
    qp, ∇qp = Qp
    ∂log_h(intersection_points, p, bj) + ∂log_qe(qp, ∇qp, e, bj)
end




### Computing Second Derivatives ###

function ∂2log_h(
    intersection_points::AbstractVector{<:AbstractVector{<:Number}},
    p::AbstractVector{<:Real},
    bj::AbstractVector{<:Real},
    )
    k = length(bj)
    i = findfirst(x->abs(x) > 1e-5, bj)
    projection_points = map(s -> s[1:k], intersection_points)
    s = map(y -> (y[i] - p[i])/bj[i], projection_points)

    -sum(1/si^2 for si in s)
end

function ∂2log_qe(Qp::Tuple, e::Real, bj::AbstractVector{<:Real})
    (qp, ∇qp) = Qp
    -e * (transpose(bj)*2*bj*qp-(transpose(∇qp)*bj)^2)/qp^2
end

function ∂2log_r(
    intersection_points::AbstractVector{<:AbstractVector{<:Number}},
    Qp::Tuple,
    e::Real,
    p::AbstractVector{<:Real},
    bj::AbstractVector{<:Real},
)
    ∂2log_h(intersection_points, p, bj) + ∂2log_qe(Qp, e, bj)
end

function compute_off_diag(intermediate, bi_val, bj_val)
    (intermediate - bi_val - bj_val)/2
end

# The following allows you to be lazy and not input a System.
hess_log_r(F::Vector{Expression}, e::Real, k::Int; kwargs...) = hess_log_r(System(F), e, k; kwargs...)
hess_log_r(F::System, e::Real, k::Int; kwargs...) = hess_log_r(F, e, variables(F)[1:k]; kwargs...)
# The following function will determine the method of computing the hessian you want and send it off.
function hess_log_r(
    F::System,
    e::Real,
    projection_vars::Vector{Variable};
    method::Symbol = :off_diag,
    c::Union{AbstractVector{<:Real}, Nothing} = nothing,
    B::Union{Matrix{<:Real}, Nothing} = nothing
)
    all_vars = variables(F)
    x_vars = setdiff(all_vars, projection_vars)
    F0 = System(F.expressions, variables = [projection_vars; x_vars])
    k = length(projection_vars)
    u = rand(ComplexF64, k)
    v = rand(ComplexF64, k)
    PWS = PseudoWitnessSet(F0, create_line(u, v, size(F0, 2)))

    if method == :off_diag
        hess = hess_log_r(PWS, e, k; c, B)
    elseif method == :many_slices
        hess = _many_slices(F, PWS, e, projection_vars; c, B)
    elseif method == :single_slice
        hess = _single_slice(F, PWS, e, projection_vars; c, B)
    end
    hess
end
#If you only have a pseudowitness set, then only one method remains.
function hess_log_r(
    PWS::PseudoWitnessSet,
    e::Real,
    k::Int;
    c::Union{AbstractVector{<:Real}, Nothing} = nothing,
    B::Union{Matrix{<:Real}, Nothing} = nothing
)
    if isnothing(c)
        c = randn(k)
    end
    if isnothing(B)
        B = Matrix(qr(randn(k,k)).Q)
    end
    function f(p)
        Qp = (sum((p - c) .^ 2) + 1, 2 .* (p - c))

        line_hypersurface_intersections = track_pws_to_lines(p, B, PWS)
        diagonals = map(zip(line_hypersurface_intersections, eachcol(B))) do (intersection_points, bj)
                ∂2log_r(intersection_points, Qp, e, p, bj)
        end
        H = diagm(diagonals)
        for i in 1:n
            for j in i+1:n
                intersection_points = track_pws_to_lines(p, B[:,i] - B[:,j], PWS)
                intermediate = ∂2log_r(intersection_points[1], Qp, e, p, B[:,i] - B[:,j])
                H[i,j] = compute_off_diag(intermediate, diagonals[i], diagonals[j])
                H[j,i] = compute_off_diag(intermediate, diagonals[i], diagonals[j])
            end
        end
        H
    end
    f
end

function _many_slices(
    F::System,
    PWS::PseudoWitnessSet,
    e::Real,
    projection_vars::Vector{Variable};
    c::Union{AbstractVector{<:Real}, Nothing} = nothing,
    B::Union{Matrix{<:Real}, Nothing} = nothing
)
# TODO: I am implementing this right now, will update soon.
end

function _single_slice(
    F::System,
    PWS::PseudoWitnessSet,
    e::Real,
    projection_vars::Vector{Variable};
    c::Union{AbstractVector{<:Real}, Nothing} = nothing,
    B::Union{Matrix{<:Real}, Nothing} = nothing
)
# TODO: This needs to be implemented! For now, redirect to the off_diag method.
hess_log_r(PWS, e, k; c, B)
end


# This function takes in a system F and the dimension k that we are projecting onto (π: \mathbb{C}^n \to \mathbb{C}^k), and returns the jacobian of F(p+s*b; u) with respect to p and the jacobian of F(p+s*b; u) with respect to s and u.
# Returns: Jacobian of F with respect to s,u, -(Jacobian) with respect to the p's, and the variables involved. 
function get_linear_system(
    F::System,
    k::Int)
    @var u, s, p[1:k], b[1:k]
    Fp = System(F([p+s*b; u]), 
    variables = vcat(p[:]),
    parameters = vcat(b[:],s,u))
    Fsu = System(F([p+s*b; u]), 
    variables = vcat(s,u),
    parameters = vcat(p[:],b[:]))
    return (jacobian(Fsu), -jacobian(Fp), [u, s, p[1:k], b[1:k]])
end