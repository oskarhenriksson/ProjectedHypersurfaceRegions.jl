using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations
include("../src/functions.jl")
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

@time hess_off_diag_eval = hess_off_diag(P)
@time hess_many_slices_eval = hess_many_slices(P) 
@time hess_single_slice_eval = hess_single_slice(P) 

norm(hess_off_diag_eval - hess_single_slice_eval) # should be small, and is about 1e-7