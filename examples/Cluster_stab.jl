using Pkg
Pkg.activate("..")
include("../src/functions.jl");

@var κ[1:6] x[1:2] c[1:2]
# κ chemical reaction rate coefficients
# c conservation constants
# x chemical concentrations
Clus_Eq = [x[1]+x[2]-c[1]-c[2], 
-κ[1]*x[1]^3*x[2]-2*κ[2]*x[1]^3*x[2]-3*κ[3]*x[1]^3*x[2]-κ[4]*x[1]^3*x[2]^2-2*κ[5]*x[1]^2*x[2]^2-κ[6]x[1]*x[2]^3]
# Equations of Cluster Stabilization Model 
# Example 3.1 from https://www.sciencedirect.com/science/article/pii/S0196885821000920

Jac = differentiate(Clus_Eq, [x[1], x[2]])

dJ = det(Jac)

F = System([Clus_Eq; dJ], variables = [κ; c; x])
projection_vars = [κ; c[1]; c[2]]  # These are the variables we are projecting to.



k = 8
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)
e = 6 # e should be even and larger than floor(degree(PWS)/2) + 1. 

hess_off_diag = hess_log_r(F_ordered, e, projection_vars; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F_ordered, e, projection_vars; method = :many_slices, c, B)

p = rand(k)
hess_off_diag(p)
hess_many_slices(p)