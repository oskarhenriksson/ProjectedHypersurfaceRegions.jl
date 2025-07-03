# Looking at the lossless four bus system, with zero power injections.



using Pkg
Pkg.activate("..")
include("../src/functions.jl");

# In this lossless four bus system with zero power injections, we have 6 parameters (6 bij where i\neq j) and 6 variables (Vd[1:3], Vq[1:3])
@var b[1:4, 1:4] Vd[1:4] Vq[1:4]
#Vd is real component of voltage
#Vq is imaginary component of voltage
#b the susceptance of the line connecting buses i and k
eqn2_2a =
    sum.([[b[i, k] * (Vd[k] * Vq[i] - Vd[i] * Vq[k]) for k = 2:4 if i != k] for i = 2:4]) + [Vq[i] * b[1, i] for i = 2:4]
bus_eqns = [
    eqn2_2a
    Vd[2:4] .^ 2 - Vq[2:4] .^ 2 - ones(3)
]
Jac = differentiate(bus_eqns, [Vd[2:4]; Vq[2:4]])
D = det(Jac)

F = System([bus_eqns; D])
# System for the incidence variety of the discriminant
all_vars = variables(F)
x_vars = [Vd[2:4]; Vq[2:4]]
projection_vars = setdiff(all_vars, x_vars)
F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
k = length(projection_vars)
PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1)

d = degree(PWS)
e = 266 # e should be even and at least  floor(degree(PWS)/2) + 1, in this case its 526
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)

hess_off_diag = hess_log_r(F_ordered, e, projection_vars; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F_ordered, e, projection_vars; method = :many_slices, c, B)

P = rand(k)
hess_off_diag_eval = hess_off_diag(P)
hess_many_slices_eval = hess_many_slices(P)

# I guess the off_diag method is much slower than many_slices when the degree of the PWS is large.
#@time hess_off_diag(P) # 18 seconds (on my machine)
#@time hess_many_slices(P) # 6 seconds (on my machine)

# Unfortunately, these evaluations are really far apart. 
# You can see something similar when you look at discriminants of deg n univariate polynomials when n gets large

hess_off_diag_eval - hess_many_slices_eval

# I think there is numerical instability.

# Checking for symmetry of Hessians
hess_many_slices_eval' - hess_many_slices_eval
hess_off_diag_eval' - hess_off_diag_eval
