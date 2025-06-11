using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra

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

# here are our two methods
p = rand(2)
hess_off_diag(p)
hess_many_slices(p)

# if you just pass in the discriminant, we can compute the Hessian directly. 
# (So this is definitely correct)
disc = a^2 - 4*b
actual_hess = hess_log_r(disc, e; c)
actual_hess(p)

# the hessians above are computed in the B-basis
B*actual_hess(p)*B^(-1) 
hess_off_diag(p)
# This is very close to 