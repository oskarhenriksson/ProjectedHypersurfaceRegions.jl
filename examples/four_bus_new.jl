# The lossless four bus system, with zero power injections.
using Pkg
Pkg.activate(".")
include("../src/functions.jl");

# In this lossless four bus system with zero power injections, we have 6 parameters (6 aij where i\neq j) and 6 variables (Vd[1:3], Vq[1:3])
# Vd is real component of voltage
# Vq is imaginary component of voltage
# a the susceptance of the line connecting buses i and k (called b in the paper)
@var a[1:4, 1:4] Vd[1:4] Vq[1:4]
part_2a = [sum([a[i, k] * (Vd[k] * Vq[i] - Vd[i] * Vq[k]) for k = 2:4 if i != k]) + Vq[i] * a[1, i] for i = 2:4]
part_2b = [Vd[i]^2 + Vq[i]^2 - 1 for i = 2:4]
bus_eqns = [part_2a; part_2b]

# Note that a[i,k] == a[k,i]. As such, only keep one. 
bus_eqns = subs(bus_eqns,
    a[2, 1] => a[1, 2], a[3, 1] => a[1, 3], a[4, 1] => a[1, 4],
    a[3, 2] => a[2, 3], a[4, 2] => a[2, 4], a[4, 3] => a[3, 4]
)
bus_eqns = expand.(bus_eqns)

# Jacobian
Jac = differentiate(bus_eqns, [Vd[2:4]; Vq[2:4]])
D = expand(det(Jac))

# Form the system and order the variables
F = System([bus_eqns; D])
all_vars = variables(F)
x_vars = [Vd[2:4]; Vq[2:4]]
projection_vars = setdiff(all_vars, x_vars)
k = length(projection_vars)
F_ordered = System(F.expressions, variables=[projection_vars; x_vars])

# Pseudo witness set
# ISSUE: This is slightly unstable. Most of the time we get d=72, 
# but sometimes we lose paths and get a lower value of d.
PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim=k-1)
d = degree(PWS) 
e = Int(floor(d / 2) + 1)

# For testing purposes, we set e to zero
e = 0

# Fix a c (to make the Hessians comparable)
c = 10 .* rand(k)


# Random point to evaluate the Hessian at
P = rand(k)

# Testing our hessian methods

# ISSUE: This gives NaN entries!
hess_off_diag = hess_log_r(F_ordered, e, projection_vars; method=:off_diag, c=c)
hess_off_diag_eval = hess_off_diag(P) 

# ISSUE: Numerically unstable. Sometimes give ArgumentError: matrix contains Infs or NaNs.
hess_many_slices = hess_log_r(F_ordered, e, projection_vars; method=:many_slices, c=c)
hess_many_slices_eval = hess_many_slices(P)

# ISSUE: Numerically unstable.
hess_single_slice = hess_log_r(F_ordered, e, projection_vars; method=:single_slice, c=c)
hess_single_slice_eval = hess_single_slice(P) 

# Comparision of the different methods
#norm(hess_off_diag_eval - hess_many_slices_eval)
norm(hess_many_slices_eval - hess_single_slice_eval) # ISSUE: Far from zero! 


########################

# Alternative formulation (avoiding computing the determinant)
# This takes longer to run (a lot of paths to track!)
m = size(Jac, 1)
@var λ[1:m]
F_ordered_alt = System([bus_eqns; Jac * λ; rand(m) ⋅ λ - 1], variables=[projection_vars; x_vars; λ])
PWS_alt = PseudoWitnessSet(F_ordered_alt, k; linear_subspace_codim=k - 1)
d_alt = degree(PWS_alt)
e_alt = Int(floor(d_alt / 2) + 1)

# For testing purposes, we set e_alt to zero 
# This ensures that the two formulations are comparable, despite the different d values
e_alt = 0

# Test the different Hessian methods for the alternative formulation
# ISSUE: Sometimes gives NaN entires.
hess_off_diag_alt = hess_log_r(F_ordered_alt, e_alt, projection_vars; method=:off_diag, c=c) 
hess_off_diag_eval_alt = hess_off_diag_alt(P) 

# ISSUE: Sometimes give ArgumentError: matrix contains Infs or NaNs.
hess_many_slices_alt = hess_log_r(F_ordered_alt, e_alt, projection_vars; method=:many_slices, c=c) 
hess_many_slices_eval_alt = hess_many_slices_alt(P) 

# Seems stable
hess_single_slice_alt = hess_log_r(F_ordered_alt, e_alt, projection_vars; method=:single_slice, c=c)
hess_single_slice_eval_alt = hess_single_slice_alt(P)

# Comparision of the different methods
norm(hess_off_diag_eval_alt - hess_many_slices_eval_alt)
norm(hess_many_slices_eval_alt - hess_single_slice_eval_alt)

# Comparison of the different formulations
norm(hess_many_slices_eval_alt - hess_many_slices_eval)
norm(hess_single_slice_eval - hess_single_slice_eval_alt)