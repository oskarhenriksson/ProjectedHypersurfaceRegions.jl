using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations

include("../grad_r_p.jl")

@var k[1:6] xy xz cy cz
# k chemical reaction rate coefficients
# c conservation constants
# x chemical concentrations
Clus_Eq = [xy+xz-cy-cz, 
-k[1]*xy^3*xz-2*k[2]*xy^3*xz-3*k[3]*xy^3*xz-k[4]*xy^3*xz^2-2*k[5]*xy^2*xz^2-k[6]xy*xz^3]
# Equations of Cluster Stabilization Model 
# Example 3.1 from https://www.sciencedirect.com/science/article/pii/S0196885821000920

Jac = differentiate(Clus_Eq, [xy, xz])

dJ = det(Jac)

F = System([Clus_Eq; dJ], variables = [k[:]; cy; cz; xy; xz])
myk = 8
B = qr(rand(myk, myk)).Q |> Matrix
c = 10 .* randn(myk)
e = 6 # e should be even and larger than floor(degree(PWS)/2) + 1. 
projection_vars = [k[:]; cy; cz]  # These are the variables we are projecting to.


hess_off_diag = hess_log_r(F_ordered, e, projection_vars; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F_ordered, e, projection_vars; method = :many_slices, c, B)

p = rand(myk)
hess_off_diag(p)
hess_many_slices(p)