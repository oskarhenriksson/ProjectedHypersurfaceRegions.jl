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

#single slice method is slower on this example. 
@time hess_off_diag_eval = hess_off_diag(P) #1.15 seconds on first run, 0.00096 seconds on second run (on my machine)
@time hess_single_slice_eval = hess_single_slice(P) #3.07 seconds on first run, 0.00424 on second run (on my machine)
@time hess_many_slices_eval = hess_many_slices(P) #0.97 seconds on first run, 0.0018 seconds on second run (on my machine)
@time hess_given_h_eval = hess_given_h(P)

norm(hess_given_h_eval - hess_single_slice_eval) # should be small, and is about 1e-14


###############################################################################################################
###############################################################################################################

#Testing on larger examples (cluster stabilization)
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

#on this example, single slice is faster than off_diag and many_slices. but it still allocates more memory than off_diag.
@time H_od = hess_off_diag(p) #0.219 seconds (on my machine), 13.84k allocations
@time H_ms = hess_many_slices(p) #0.085 seconds (on my machine), 108.03k allocations 
@time H_ss = hess_single_slice(p) #0.051 seconds (on my machine) 51.34k allocations
norm(H_od - H_ss) #still very small, on the order of 1e-12

###############################################################################################################
###############################################################################################################


#Example host-range expansion. 3 dimensional system from this paper: https://epubs.siam.org/doi/10.1137/23M1605582


#Setting up the system (cleared denominators) here we have that κ_i=1/K_i where K_i is the carrying capacity of species i
@var N[1:2] P r[1:2] κ[1:2] β λ 

DE_eqs = [r[1]*N[1]*(1-N[1]*κ[1])-N[1]*N[2]-N[1]*P,
          r[2]*N[2]*(1-N[1]*κ[1])-N[2]*N[1],
          β*N[1]*P-P
    ]

#adding the condition that solutions to the system are singular
J=differentiate(vcat(DE_eqs),vcat(N,P))

#getting the characteristic polynomial 
char=det(λ*I-J)


#getting the coeffs of the characteristic poly 
#to use in the Routh array

coeffs=coefficients(char,λ)

#Explicitly constructing the Hurwitz matrix
#(there is definitely a more clever way to do this)

hurwitz = zeros(Expression, 3, 3)

a_3=coeffs[1]
a_2=coeffs[2]
a_1=coeffs[3]
a_0=coeffs[4]

hurwitz[1,1]=a_2
hurwitz[1,2]=a_0 
hurwitz[2,1]=a_3 
hurwitz[2,2]=a_1
hurwitz[3,2]=a_2
hurwitz[3,3]=a_0


#Routh_Hurwitz polynomials are the principal minors of the Hurwitz matrix
minors = Expression[]
for i in 1:3
    minor = det(hurwitz[1:i, 1:i])
    push!(minors, minor)
end

RH_boundary_eqs = minors[1]*minors[3]

#System for the incidence variety of the hypersurface
F = System(vcat(DE_eqs,RH_boundary_eqs), variables = vcat(β, r, κ, N, P))

projection_variables = [β; r; κ]
k = length(projection_variables)

B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)
F_ordered = System(F.expressions, variables = [projection_variables; N; P])

PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1)

e = Int(floor(degree(PWS) / 2)) + 1

#testing hessian methods

hess_off_diag = hess_log_r(F_ordered, e, projection_variables; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F_ordered, e, projection_variables; method = :many_slices, c, B)
hess_single_slice = hess_log_r(F_ordered, e, projection_variables; method = :single_slice, c, B)

P = rand(k)

@time hess_off_diag_eval = hess_off_diag(P) #around 0.19 seconds (on my machine), 9.47k allocations
@time hess_many_slices_eval = hess_many_slices(P) #around 0.09 seconds (on my machine), 216.49k allocations
@time hess_single_slice_eval = hess_single_slice(P) #around 0.22 seconds (on my machine), 434.48k allocations

norm(hess_off_diag_eval - hess_single_slice_eval) # should be small, and is about 1e-7

#######################################################################################
#######################################################################################

# Another example, this time the four bus example 
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
#Note that b[i,k] == b[k,i]. As such, only keep one. 
bus_eqns = subs(bus_eqns, b[2,1]=> b[1,2], b[3,1]=> b[1,3], b[4,1]=> b[1,4], b[3,2]=> b[2,3], b[4,2]=> b[2,4], b[4,3]=> b[3,4])
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
e= Int(floor(d/2)+1)
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)
######################################################
#Testing our hessian methods
hess_off_diag = hess_log_r(F_ordered, e, projection_vars; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F_ordered, e, projection_vars; method = :many_slices, c, B)
hess_single_slice = hess_log_r(F_ordered, e, projection_vars; method = :single_slice, c, B)


#This is an example where the mysterious bug mentioned in the TODO list occurs. 
#The optimized single_slice method (using caching) is subject to this bug as well. 
P = rand(k)
@time hess_off_diag_eval = hess_off_diag(P)
@time hess_many_slices_eval = hess_many_slices(P)
@time hess_single_slice_eval = hess_single_slice(P)