# Ecological system from https://www.worldscientific.com/doi/abs/10.1142/S0218127421502023
# Unknown computation time

using Pkg, Random
Pkg.activate(".")

include("../src/functions.jl");

Random.seed!(0x8b868320)

### Set up incidence variety of discriminant
@var a b x[1:3]
steady_state = [x[1]*(1 - x[1])*(x[1] - b) - 2*a*x[1] + a*x[2] + a*x[3], 
x[2]*(1 - x[2])*(x[2] - b) - 2*a*x[2] + a*x[1] + a*x[3], 
x[3]*(1 - x[3])*(x[3] - b) - 2*a*x[3] + a*x[1] + a*x[2]]
Jac = differentiate.(steady_state, x')
detJac = det(Jac)
F = System([steady_state; x[1]*x[2]*x[3]*detJac], variables = [a; b; x])

### Routing gradient
r = RoutingGradient(F, [a,b]);
d = degree(r.PWS) 

### Homotopy
k = size(r, 2) # number of variables
p1 = zeros(k)
q1 = randn(k)
H = RoutingPointsHomotopy(r, p1, q1)

### Solve system using monodromy

# Set up monodromy solver
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

# Start pair
s0 = randn(ComplexF64, k)
p0 = evaluate(r, s0)

# Solve r = p0
seed = rand(UInt32)
mon_result = monodromy_solve(
    MS,
    s0,
    p0,
    seed
)

# Trace to solution of r = 0
intermediate_p = randn(ComplexF64, length(p0))
start_parameters!(H, p0)
target_parameters!(H, intermediate_p)
result_intermediate = HomotopyContinuation.solve(H, solutions(mon_result))
start_parameters!(H, intermediate_p)
target_parameters!(H, zeros(ComplexF64, length(p0)))
result = HomotopyContinuation.solve(H, result_intermediate)

# Extract the real solutions
pts = real_solutions(result)

### Connected components
G, idx, failed_info = partition_of_critical_points(r, pts)