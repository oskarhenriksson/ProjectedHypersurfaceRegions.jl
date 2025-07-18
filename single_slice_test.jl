using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations, Plots

include("./src/functions.jl")

### testing the single slice method

@var x a b
F = System([x^2 + a*x + b; 2*x + a], variables = [a, b, x])

projection_variables = [a; b]

c = 10 .* randn(2)

k = length(projection_variables)

B = qr(rand(k, k)).Q |> Matrix

e = 2

hess_off_diag = hess_log_r(F, e, projection_variables; method = :off_diag, c, B)

hess_many_slices = hess_log_r(F, e, projection_variables; method = :many_slices, c, B)

hess_single_slice = hess_log_r(F, e, projection_variables; method = :single_slice, c, B)

h = a^2-4*b
hess_given_h = hess_log_r_given_h(h,e;c)
P = rand(k)

#the single slice method is currently inaccurate and allocates a lot of memory
#we will have to optimize this.  
@time hess_off_diag_eval = hess_off_diag(P)
@time hess_single_slice_eval = hess_single_slice(P)
@time hess_many_slices_eval = hess_many_slices(P)
@time hess_given_h_eval = hess_given_h(P)