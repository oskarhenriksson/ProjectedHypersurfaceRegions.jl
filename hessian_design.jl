# This file is work in progress towards coding up Jon's suggestion to compute the Hessian with a bunch of linear solves. It seems to work for our example.


using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, Plots, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations

include("grad_r_p.jl")


@var a b x
F = System([x^2 + a*x + b; 2x + a], variables = [a, b, x])
k = 2
# System for the incidence variety of the discriminant

# B is a orthonormal basis, which we will use to create two lines in direction b_1 and b_2 (b_i are columns of B) 
B = qr(rand(2, 2)).Q |> Matrix

#c and e are parameters for creating the denominator of our routing function r(p) = h(p)/q^e
c = 10 .* randn(2)
e = 2

###### Critical points 
pts = include("discr_pw.jl")
#These our p_i in our computations. This is hard coded for the discriminant variety for now.

###### PWS
PWS = PseudoWitnessSet(F, create_line(rand(ComplexF64, 2), rand(ComplexF64, 2), 3))
# These are our u_i's. 

#lines will track the random psuedowitness set to the lines defined by b
#it will have two different solutions, one for each column of B (for each line)
lines = track_pws_to_lines(pts[1], B, PWS)
#line 1 corresponds to the first column of B, and has two 3 dimensional vectors. These are the u's above the intersection.

#But now we need to compute the t such that p+t*b=s_i. s_1 is lines[1][1][1:2]
(lines[1][1][1:2][1] - pts[1][1])./ B[1,:][1]

(X,Y,system_vars) = get_linear_system(F, 2) 

u, s, p, b = system_vars
Xeval = evaluate(X, vcat(p[:],b[:],s,u) => vcat(
    pts[1],
    B[1,:],
    (lines[1][1][1:2][1] - pts[1][1])./ B[1,:][1],
    lines[1][1][3]
) )
Yeval = evaluate(Y, vcat(p[:],b[:],s,u) => vcat(
    pts[1],
    B[1,:],
    (lines[1][1][1:2][1] - pts[1][1])./ B[1,:][1],
    lines[1][1][3]
) )


sols = Xeval \ Yeval
## Need to evaluate Jac_p(-sum_(i=1)^d 1/s_k)

## Need to add in the hardcoded term corresponding to the q term.
## eJac_p(q)/q

#########################################
#This is Hannah's solution
hess = hess_log_r(2, 2, PWS)

hess(rand(2))