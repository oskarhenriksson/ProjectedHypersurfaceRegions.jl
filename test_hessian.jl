using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations

include("grad_r_p.jl")


@var a b x
F = System([x^2 + a*x + b; 2x + a], variables = [a, b, x])
# System for the incidence variety of the discriminant

B = qr(rand(2, 2)).Q |> Matrix
c = 10 .* randn(2)
e = 2
k = 2

###### Critical points 
#pts = include("discr_pw.jl")

hess_off_diag = hess_log_r(F, e, k; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F, e, k; method = :many_slices, c, B)

# You can also call hess_log_r(F, e, k; method = :off_diag, c, B)
# c and B are optional -- otherwise, they are taken randomly.

# if you just pass in the discriminant, we can compute the Hessian directly. 
# (So this is definitely correct)
disc = a^2 - 4*b
actual_hess = hess_log_r(disc, e; c)


p = rand(2)
actual_hess(p)
hess_off_diag(p)
hess_many_slices(p)

@time for _=1:1000 p = rand(2); hess_off_diag(p) end

@time for _=1:1000 p = rand(2); hess_many_slices(p) end

## A 3-dimension discriminant example

@var α β γ x
F = System([x^3 + α*x^2 + β*x + γ; 3*x^2 + 2*α*x + β], variables = [α, β, γ, x])
# System for the incidence variety of the discriminant

B = qr(rand(3, 3)).Q |> Matrix
c = 10 .* randn(3)
e = 4
k = 3


hess_off_diag = hess_log_r(F, e, k; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F, e, k; method = :many_slices, c, B)

disc = α^2*β^2 - 4*β^3 - 4*α^3*γ - 27*γ^2 + 18*α*β*γ
actual_hess = hess_log_r(disc, e; c)

# These are all the same!
p = rand(3)
hess_off_diag(p)
hess_many_slices(p)
@time actual_hess(p)

# Time analysis
@time for _=1:1000 p = rand(3); hess_off_diag(p) end
@time for _=1:1000 p = rand(3); hess_many_slices(p) end