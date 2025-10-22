# Example from thermodynamics
# This is a discriminant that controls the possible geometries that a spinodal curve can take
# in a phase diagram for a system with two species and no solvent-interactions

# Terminated within a couple of hours but gave incomplete solution set

using Pkg, Random
Pkg.activate(".")

include("../src/functions.jl");

Random.seed!(0x8b868320)

# The parameters of the model (interaction coefficients)
@var e11 e12 e22
ε = [e11, e12, e22]

# The variables of the model (concentrations of the two species)
@var x[1:2]

# Polynomial defining the spinodal curve
h = -4 * e11 * e22 * x[1]^2 * x[2] - 4 * e11 * e22 * x[1] * x[2]^2 + 
        4 * e12^2 * x[1]^2 * x[2] + 4 * e12^2 * x[1] * x[2]^2 + 
        4 * e11 * e22 * x[1] * x[2] - 4 * e12^2 * x[1] * x[2] + 
        2 * e11 * x[1]^2 + 4 * e12 * x[1] * x[2] + 
        2 * e22 * x[2]^2 - 2 * e11 * x[1] - 2 * e22 * x[2] + 1

# Set up incidence variety for spinodal discriminant
F = System([h, differentiate(h, x[1]), differentiate(h, x[2])], variables=[ε; x])

# Routing gradient
r = RoutingGradient(F, ε)

# This gives 6 but I know the discriminant has degree 72
d = degree(r.PWS) 

# Critical points

### Homotopy
k = size(r, 2) # number of variables
p1 = zeros(k)
q1 = randn(k)
H = RoutingPointsHomotopy(r, p1, q1)

### Use monodromy to solve ∇r = p0 for generic RHS
egtracker = EndgameTracker(H) # we want to add options later 
trackers = [egtracker]
x₀ = zeros(ComplexF64, size(H, k))
unique_points = UniquePoints(
    x₀,
    1;
)
trace = zeros(ComplexF64, length(x₀) + 1, 3)
P = Vector{ComplexF64}
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 10 # change the stopping criterion
)


MS = HomotopyContinuation.MonodromySolver(
    trackers,
    HomotopyContinuation.MonodromyLoop{P}[],
    unique_points,
    ReentrantLock(),
    options,
    HomotopyContinuation.MonodromyStatistics(),
    trace,
    ReentrantLock(),
)

### start pair
s0 = randn(ComplexF64, k)
p0 = evaluate(r, s0)

### monodromy 
seed = rand(UInt32)
mon_result = monodromy_solve(
    MS,
    s0,
    p0,
    seed
)

### move to parameter = 0
intermediate_p = randn(ComplexF64, length(p0))
start_parameters!(H, p0)
target_parameters!(H, intermediate_p)
result_intermediate = HomotopyContinuation.solve(H, solutions(mon_result))
start_parameters!(H, intermediate_p)
target_parameters!(H, zeros(ComplexF64, length(p0)))
result = HomotopyContinuation.solve(H, result_intermediate)

# Extract the real solutions
pts = real_solutions(result)

# Connected components
G, idx, failed_info = partition_of_critical_points(r, pts)