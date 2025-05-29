using HomotopyContinuation, LinearAlgebra

N = 3
@var a b x
F = System([x^2 + a*x + b; 2x + a], variables = [a,b,x])
nvariables(F)


B = qr(rand(2,2)).Q |> Matrix
p = [10, -10]


mutable struct PseudoWitnessSet
    F::System
    L::LinearSubspace
    W::Result
end

function PseudoWitnessSet(F::System, L::LinearSubspace)
    n = ambient_dim(L)
    startL = rand_subspace(n; codim = 1)
    S = solutions(witness_set(F, startL))
    W = solve(F, S, start_subspace = startL,
                     target_subspace = L, intrinsic=true)
    PseudoWitnessSet(F, L, W)                 
end

"""Returns a LinearSubspace u + tv in a subspace of R^n"""
function create_line(u::AbstractVector{<:Number}, 
                     v::AbstractVector{<:Number}, 
                     n::Int64)
    k = length(u)
    πA = u' - (dot(u,v)/dot(v,v)) * v'
    A = [πA zeros(n-k)]
    b = [πA * u]

    LinearSubspace(A, b)
end


function track_pws_to_lines(p::AbstractVector{<:Real}, 
                            B::Matrix{Float64}, 
                            PWS::PseudoWitnessSet)
    L = PWS.L
    Ks = map(bj -> create_line(p, bj, ambient_dim(L)), eachcol(B))
    solve(PWS.F, PWS.W, start_subspace = L,
                        target_subspaces = Ks, 
                        intrinsic = true, 
                        transform_result = (r,p) -> solutions(r))
end


function Q(p::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    y = p - c

    (sum(y.^2) + 1, 2 .* y)
end

### Computing Derivatives ###
function directional_derivative_h(intersection_points::AbstractVector{<:AbstractVector{<:Number}}, 
                                  p::AbstractVector{<:Real}, 
                                  bj::AbstractVector{<:Real})
    k = length(bj)
    i = findfirst(x->abs(x) > 1e-5, bj)
    projection_points = map(s -> s[1:k], intersection_points)
    s = map(y -> (y[i] - p[i])/bj[i], projection_points)

    -sum(1/si for si in s)
end

function directional_derivative_qe(qp::Real, 
                                   ∇qp::AbstractVector{<:Real}, 
                                   e::Real, 
                                   bj::AbstractVector{<:Real})
    -e * (transpose(∇qp) * bj)/qp
end

function directional_derivative_r(intersection_points::AbstractVector{<:AbstractVector{<:Number}}, 
                                  qp::Real, 
                                  ∇qp::AbstractVector{<:Real}, 
                                  e::Real, 
                                  p::AbstractVector{<:Real}, 
                                  bj::AbstractVector{<:Real})
    directional_derivative_h(intersection_points, p, bj) + directional_derivative_qe(qp, ∇qp, e, bj)
end

function directional_grad_r(c::AbstractVector{<:Real}, 
                            e::Real, 
                            p::AbstractVector{<:Real}, 
                            B::Matrix{<:Real}, 
                            PWS::PseudoWitnessSet)
    qp, ∇qp = Q(p, c)
    line_hypersurface_intersections = track_pws_to_lines(p, B, PWS)

    map(zip(line_hypersurface_intersections, eachcol(B))) do (intersection_points, bj)
        directional_derivative_r(intersection_points, qp, ∇qp, e, p, bj)
    end
end

function grad_r(c::AbstractVector{<:Real}, 
                e::Real, 
                p::AbstractVector{<:Real}, 
                PWS::PseudoWitnessSet)
    k = length(c)            
    B = qr(rand(k,k)).Q |> Matrix
    B * directional_grad_r(c, e, p, B, PWS) 
end

u = rand(ComplexF64, 2)
v = rand(ComplexF64, 2)
PWS = PseudoWitnessSet(F, create_line(u,v,3));
grad_r([1/3, 1/5], 2, p, PWS)


pws_sets = track_pws_to_lines(p, B[:, 1:1], PWS)

c = rand(2)
e = 2
qp, ∇qp = Q(p, c)
directional_derivative_qe(qp, ∇qp,2, B[:,1])

directional_grad_r([1/3, 1/5], 2, p, B, PWS)
