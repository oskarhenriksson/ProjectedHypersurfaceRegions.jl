function hess_log_r(disc::Expression, e::Real; c::Union{AbstractVector{<:Real}, Nothing} = nothing)
    #TODO: Implement the Hessian of the log of the discriminant.
    if isnothing(c)
        c = randn(length(variables(disc)))
    end
    p = variables(disc)
    e = 2
    q = 1 + sum((p-c) .* (p-c))
    r = log(disc/q^e)
    H = differentiate(differentiate(r,p),p)
    hess(P) = evaluate(H, p => P)
    hess
end