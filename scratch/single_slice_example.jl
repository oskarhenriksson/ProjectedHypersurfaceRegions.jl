# An attempt to implement the single slice method for computing the Hessian

using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations, Plots

include("../src/functions.jl")
include("single_slice.jl")

# Incidence variety
@var a b γ δ x y
G = [x^3 + a * x^2 + b*x*y + γ; x - y + δ]
JG = differentiate(G, [x; y]);
F = System([G; det(JG)],  variables = [a; b; γ; δ; x; y])

# Reorder the variables of F
projection_variables = [a; b; γ; δ]
x_variables = setdiff(variables(F), projection_variables)
F = System(F.expressions, variables = [projection_variables; x_variables])

# Fix a denominator point
denominator_point = [3; -2; 5; -1]

# Fixed direction
B = [2; 5; -1; 3]

# Point for evaluation
P = [1,2,3,4]

# Actual h polynomial
@var c d
h = a^2*b^2*d^2+2*a*b^3*d^2+b^4*d^2-4*b^3*d^3-4*a^3*c-12*a^2*b*c-12*a*b^2*c-4*b^3*c+18*a*b*c*d+18*b^2*c*d-27*c^2
h = h(c => γ, d => δ)
grad_symbolic = differentiate(log(h), projection_variables)
grad_true = grad_symbolic(projection_variables => P)
hess_symbolic = differentiate(grad_symbolic, projection_variables)
hess_true = hess_symbolic(projection_variables => P)

# Set up function for hess(q)
q = 1 + sum((p - denominator_point) .* (p - denominator_point))
Hlogqe(pt) = evaluate(differentiate(differentiate(log(q^e), p), p), p => pt) 

# Test of our new function
PWS = PseudoWitnessSet(F, 4, linear_subspace_codim=3)
e = Int(floor( degree(PWS)/2) + 1)

our_hessian = hess_log_r(F, e, projection_variables; method = :single_slice, c = denominator_point) 
@time H = our_hessian(P)

our_hessian_off_diag = hess_log_r(F, e, projection_variables; method = :off_diag, c = denominator_point) 
@time H_off_diag = our_hessian_off_diag(P)

norm(H' - H) # should be close to zero
norm(hess_true + Hlogqe(P) - H)  # should be close to zero
