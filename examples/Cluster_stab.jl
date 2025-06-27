using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations

include("../src/functions.jl");

@var κ[1:6] xy xz cy cz
# κ chemical reaction rate coefficients
# c conservation constants
# x chemical concentrations
Clus_Eq = [xy+xz-cy-cz, 
-κ[1]*xy^3*xz-2*κ[2]*xy^3*xz-3*κ[3]*xy^3*xz-κ[4]*xy^3*xz^2-2*κ[5]*xy^2*xz^2-κ[6]xy*xz^3]
# Equations of Cluster Stabilization Model 
# Example 3.1 from https://www.sciencedirect.com/science/article/pii/S0196885821000920

Jac = differentiate(Clus_Eq, [xy, xz])

dJ = det(Jac)

F = System([Clus_Eq; dJ], variables = [κ; cy; cz; xy; xz])
k = 8
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)
e = 6 # e should be even and larger than floor(degree(PWS)/2) + 1. 
projection_vars = [κ; cy; cz]  # These are the variables we are projecting to.


hess_off_diag = hess_log_r(F_ordered, e, projection_vars; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F_ordered, e, projection_vars; method = :many_slices, c, B)

p = rand(k)
hess_off_diag(p)
hess_many_slices(p)