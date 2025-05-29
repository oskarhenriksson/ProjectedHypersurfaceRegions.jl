using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, Plots, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations

include("grad_r_p.jl")


@var a b x
F = System([x^2 + a*x + b; 2x + a], variables = [a, b, x])
# System for the incidence variety of the discriminant

B = qr(rand(2, 2)).Q |> Matrix
c = 10 .* randn(2)
e = 2

###### Critical points 
pts = include("discr_pw.jl")

###### PWS
u = rand(ComplexF64, 2)
v = rand(ComplexF64, 2)
PWS = PseudoWitnessSet(F, create_line(u, v, 3))

p = rand(2)



hess = hess_log_r(2, 2, PWS)

hess(rand(2))