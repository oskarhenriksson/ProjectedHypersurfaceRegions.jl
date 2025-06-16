  

struct PseudoWitnessSet
    F::System
    L::LinearSubspace
    W::Result
end
degree(PWS::PseudoWitnessSet) = length(PWS.W)
ambient_dim(PWS::PseudoWitnessSet) = HC.ambient_dim(PWS.L)

function PseudoWitnessSet(F::System, L::LinearSubspace) 
    n = HC.ambient_dim(L)
    startL = rand_subspace(n; codim = 1)
    S = solutions(witness_set(F, startL))
    W = HC.solve(F, S, start_subspace = startL, target_subspace = L, intrinsic = true)
    PseudoWitnessSet(F, L, W)
end


mutable struct GradientCache
    A::Matrix
    πA::Matrix
    b::Vector
    Id::Diagonal
    v_perp::Vector
    Ks::Vector{LinearSubspace}
    s::Vector
    line_hypersurface_intersections::Vector
    intersection_points::Vector
    projection_points::Vector

end
function GradientCache(k, n, d)
    A = zeros(ComplexF64, k - 1, n)
    πA = zeros(ComplexF64, k - 1, k)
    b = zeros(ComplexF64, n)
    Id = I(k)
    v_perp = zeros(ComplexF64, k)
    Ks = Vector{LinearSubspace}(undef, k)
    s = Vector{Vector{ComplexF64}}(undef, d)
    line_hypersurface_intersections = Vector{Result}(undef, d)
    intersection_points = Vector{Vector{ComplexF64}}(undef, d)
    projection_points = Vector{Vector{ComplexF64}}(undef, d)
    

    GradientCache(A, πA, b, Id, v_perp, Ks, s,
                    line_hypersurface_intersections, 
                    intersection_points,
                    projection_points)
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
    
    n = ambient_dim(PWS)
    d = degree(PWS)
    GC = GradientCache(k, n, d)

    Q(p) = (sum((p - c) .^ 2) + 1, 2 .* (p - c))

    function f(p)
        Qp = Q(p)

        track_pws_to_lines!(GC, p, B, PWS)

        out = map(zip(GC.line_hypersurface_intersections, eachcol(B))) do (intersection_points, bj)
            ∂log_r(intersection_points, Qp, e, p, bj)
        end

        real(B * out)
    end

    return f
end





"""Returns a LinearSubspace of the form {u + tv} x R^{n-k} in R^n"""
function create_line(u::AbstractVector{<:Number}, v::AbstractVector{<:Number}, n::Int64)
    k = length(u)
    πA = randn(k-1, k) * (I(k) - (v * v')/dot(v, v))
    A = [πA zeros(k-1, n-k)]
    b = πA * u

    LinearSubspace(A, b)
end


function track_pws_to_lines!(
    GC::GradientCache,
    p::AbstractVector{<:Real},
    B::AbstractArray{Float64},
    PWS::PseudoWitnessSet,
)
    L = PWS.L
    n = ambient_dim(PWS)
    for (j,bj) in enumerate(eachcol(B))
        GC.Ks[j] = create_line(p, bj, n)
    end

    GC.line_hypersurface_intersections = HC.solve(
        PWS.F,
        PWS.W,
        start_subspace = L,
        target_subspaces = GC.Ks,
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

function hess_log_r(
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


