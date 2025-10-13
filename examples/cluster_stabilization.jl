using Pkg
Pkg.activate(".")
include("../src/functions.jl");

@var κ[1:6] x[1:2] T[1:2]
# κ chemical reaction rate coefficients
# T conservation constants (total amounts)
# x chemical concentrations
Clus_Eq = [
    x[1] + x[2] - T[1] - T[2],
    -κ[1] * x[1]^3 * x[2] - 2 * κ[2] * x[1]^3 * x[2] - 3 * κ[3] * x[1]^3 * x[2] -
    κ[4] * x[1]^3 * x[2]^2 - 2 * κ[5] * x[1]^2 * x[2]^2 - κ[6]x[1] * x[2]^3,
]
# Equations of Cluster Stabilization Model 
# Example 3.1 from https://www.sciencedirect.com/science/article/pii/S0196885821000920

Jac = differentiate(Clus_Eq, [x[1], x[2]])

dJ = det(Jac)

F = System([Clus_Eq; dJ], variables = [κ; T; x])
projection_vars = [κ; T]  # These are the variables we are projecting to.

k = 8
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)
F_ordered = System(F.expressions, variables = [projection_vars; x])
PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1)
e = Int(floor(degree(PWS) / 2)) + 1



hess_off_diag = hess_log_r(F_ordered, e, projection_vars; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F_ordered, e, projection_vars; method = :many_slices, c, B)
hess_single_slice = hess_log_r(F_ordered, e, projection_vars; method = :single_slice, c, B)
p = rand(k)
@time H_od = hess_off_diag(p)
@time H_ms = hess_many_slices(p)
@time H_ss = hess_single_slice(p)
norm(H_od - H_ss)
