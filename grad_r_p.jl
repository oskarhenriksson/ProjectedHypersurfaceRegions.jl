struct PseudoWitnessSet
    F::System
    L::LinearSubspace
    W::Result
end

function PseudoWitnessSet(F::System, L::LinearSubspace)
    n = ambient_dim(L)
    startL = rand_subspace(n; codim = 1)
    S = solutions(witness_set(F, startL))
    W = HC.solve(F, S, start_subspace = startL, target_subspace = L, intrinsic = true)
    PseudoWitnessSet(F, L, W)
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
    B::Matrix{Float64},
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

function ∇log_r(
    e::Real,
    k::Int,
    PWS::PseudoWitnessSet;
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

        out = map(zip(line_hypersurface_intersections, eachcol(B))) do (intersection_points, bj)
            ∂log_r(intersection_points, Qp, e, p, bj)
        end

        real(B* out)
    end

    return f
end


# using HomotopyContinuation, LinearAlgebra




# mutable struct PseudoWitnessSet
#     F::System
#     L::LinearSubspace
#     PWS::Result
# end

# function PseudoWitnessSet(F::System, L::LinearSubspace)
#     n = ambient_dim(L)
#     startL = rand_subspace(n; codim = 1)
#     W = witness_set(F, startL)
#     S = solutions(W)
#     PWS = HC.solve(F, S, start_subspace = startL,
#                      target_subspace = L, intrinsic=true)
#     PseudoWitnessSet(F, L, PWS)                 
# end

# """Returns a LinearSubspace u + tv in a subspace of R^n"""
# function create_line(u, v, n)
#     k = length(u)
#     πA = u' - (dot(u,v)/dot(v,v)) * v'
#     A = [πA zeros(n-k)]
#     B = [πA * u]

#     LinearSubspace(A, B)
# end



# function track_pws_to_lines(p, B, PWS)
#     L = PWS.L
#     n = ambient_dim(L)
#     Ks = map(eachcol(B)) do bj
#         create_line(p, bj, n)
#     end
#     HC.solve(PWS.F, PWS.PWS, start_subspace = L,
#                         target_subspaces = Ks, intrinsic = true, 
#                         transform_result = (r,p) -> solutions(r))
# end


# function Q(p, c)
#     y = p - c

#     (sum(y.^2) + 1, 2 .* y)
# end

# ### Computing Derivatives ###
# function directional_derivative_h(pws, p, bj)
#     k = length(bj)
#     i = findfirst(x->abs(x) > 1e-5, bj)
#     Y = map(s -> s[1:k], pws)
#     s = map(Y) do y
#         (y[i] - p[i])/bj[i]
#     end

#     -sum(1/si for si in s)
# end
# function directional_derivative_qe(qp, ∇qp, e, bj)
#     -e * (transpose(∇qp) * bj) / qp
# end
# function directional_derivative_r(pws, qp, ∇qp, e, p, bj)
#     directional_derivative_h(pws, p, bj) + directional_derivative_qe(qp, ∇qp, e, bj)
# end

# # directional_derivative_r(pws_sets[1], [1/3, 1/5, 1/7],2, p, B[:,1])


# function directional_grad_r(c, e, p, B, PWS)
#     qp, ∇qp = Q(p, c)
#     pws_for_lines = track_pws_to_lines(p, B, PWS)

#     map(zip(pws_for_lines, eachcol(B))) do (pws, bj)
#         directional_derivative_r(pws, qp, ∇qp, e, p, bj)
#     end
# end
