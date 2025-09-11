# Quick comparison test between system.jl and slice_system.jl
import Pkg; Pkg.activate(".")
using HomotopyContinuation;

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
e=2
# Construct both RoutingGradient objects
r_sys = SysImpl.RoutingGradient(F, projection_vars; B = B, c = c)
r_slice = SliceImpl.RoutingGradient(F, projection_vars; B = B, c = c, e = e)


p1 = zeros(2) # starting point 
q1 = randn(2) # target point
H_sys = SysImpl.RoutingPointsHomotopy(r_sys, p1, q1)
H_slice = SliceImpl.RoutingPointsHomotopy(r_slice, p1, q1)
# These are homotopies grad(r(x)) - t*p_1 - (1-t)*q_1
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
# So evaluate works.


# compute jacobians
u1, J1 = SysImpl.evaluate_and_jacobian(r_sys, u)
u2, J2 = SliceImpl.evaluate_and_jacobian(r_slice, u)
println("max |u1-u2| = ", maximum(abs.(u1-u2)))
println("max |J1-J2| = ", maximum(abs.(J1 - J2)))
# So jacobian works.

### Test monodromy

egtracker_sys = EndgameTracker(H_sys) # we want to add options later 
egtracker_slice = EndgameTracker(H_slice)
trackers_sys = [egtracker_sys]
trackers_slice = [egtracker_slice]
x₀ = zeros(ComplexF64, size(H_sys, 2))

unique_points = UniquePoints(
    x₀,
    1;
)

trace = zeros(ComplexF64, length(x₀) + 1, 3)
P = Vector{ComplexF64}
options = MonodromyOptions()
MS_sys = HomotopyContinuation.MonodromySolver(
    trackers_sys,
    HomotopyContinuation.MonodromyLoop{P}[],
    unique_points,
    ReentrantLock(),
    options,
    HomotopyContinuation.MonodromyStatistics(),
    trace,
    ReentrantLock(),
)
MS_slice = HomotopyContinuation.MonodromySolver(
    trackers_slice,
    HomotopyContinuation.MonodromyLoop{P}[],
    unique_points,
    ReentrantLock(),
    options,
    HomotopyContinuation.MonodromyStatistics(),
    trace,
    ReentrantLock(),
)

#### set up start pair
s0 = randn(ComplexF64, 2)
p0_sys = evaluate(r_sys, s0)
p0_slice = evaluate(r_slice, s0)
# so p0_sys == p0_slice!

evaluate(r_slice, s0, p0_slice) # equals zero
evaluate(r_sys, s0, p0_sys) # equals zero

### Monodromy 
seed = rand(UInt32)
mon_result_sys = monodromy_solve(
    MS_sys,
    s0,
    p0_sys,
    seed;
)
mon_result_slice = monodromy_solve(
    MS_slice,
    s0,
    p0_slice,
    seed;
)

# mon_result_slice gives a warning that none of the solutions is a valid start solution.
# This happens because check_start_solutions returns an empty PathResult[]:
HomotopyContinuation.check_start_solutions(MS_slice, [s0], p0)

# Digging in further, check_start_solutions fails when it runs the following:
tracker = MS.trackers[1]
parameters!(tracker, p0, p0)
track(tracker, s0) # terminated, invalid start value

