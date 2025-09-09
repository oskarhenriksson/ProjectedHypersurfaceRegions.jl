# Quick comparison test between system.jl and slice_system.jl
import Pkg; Pkg.activate(".")
using HomotopyContinuation;
# load helpers
# load the two routing gradient implementations under different module names by
# using `include` into local modules to avoid name clashes
module SysImpl
using ..HomotopyContinuation
const HC = HomotopyContinuation
include("src/pseudo_witness_sets.jl")
include("src/gradient_hessian.jl")
include("src/system.jl")
include("src/homotopy.jl")
end
module SliceImpl
using ..HomotopyContinuation
const HC = HomotopyContinuation
include("src/pseudo_witness_sets.jl")
include("src/gradient_hessian.jl")
include("src/slice_system.jl")
include("src/homotopy.jl")
end
using LinearAlgebra, Random

# Build a small test system similar to homotopy_test.jl
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])
projection_vars = [a, b]


B = qr(rand(2, 2)).Q |> Matrix
c = 10 .* randn(2)
# Construct both RoutingGradient objects
r_sys = SysImpl.RoutingGradient(F, projection_vars; B = B, c = c)
r_slice = SliceImpl.RoutingGradient(F, projection_vars; B = r_sys.B, c = r_sys.c, e = r_sys.e)


p1 = zeros(2)
q1 = randn(2)
H_sys = SysImpl.RoutingPointsHomotopy(r_sys, p1, q1)
H_slice = SliceImpl.RoutingPointsHomotopy(r_slice, p1, q1)
# pick a random x
u = randn(ComplexF64, 2)
x0 = randn(2)
t0 = 1.0
SysImpl.evaluate!(u, H_sys, x0, t0)
u
SliceImpl.evaluate!(u, H_slice, x0, t0)
u

u1 = randn(ComplexF64, 2)
t1 = 0.0
SysImpl.evaluate!(u1, H_sys, x0, t1)
u1
SliceImpl.evaluate!(u1, H_slice, x0, t1)
u1



# compute jacobians
u1, J1 = SysImpl.evaluate_and_jacobian(r_sys, u)
u2, J2 = SliceImpl.evaluate_and_jacobian(r_slice, u)
println("max |u1-u2| = ", maximum(abs.(u1-u2)))
println("max |J1-J2| = ", maximum(abs.(J1 - J2)))
