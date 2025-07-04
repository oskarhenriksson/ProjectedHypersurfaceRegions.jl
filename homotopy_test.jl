using Pkg
Pkg.activate(".")

using Plots, DifferentialEquations

include("src/functions.jl");

########
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])


B = qr(rand(2, 2)).Q |> Matrix
c = 10 .* randn(2)
r = RoutingGradient(F, [a; b]; c = c, B = B)

p1 = zeros(2)
q1 = randn(2)
H = RoutingPointsHomotopy(r, p1, q1)


### Test evaluation
u = randn(ComplexF64, 2)
x0 = randn(2)
t0 = 1.0
evaluate!(u, H, x0, t0)

u1 = randn(ComplexF64, 2)
t1 = 0.0
evaluate!(u1, H, x0, t1)

### Test monodromy

egtracker = EndgameTracker(H) # we want to add options later 
trackers = [egtracker]
x₀ = zeros(ComplexF64, size(H, 2))

unique_points = UniquePoints(
    x₀,
    1;
)

trace = zeros(ComplexF64, length(x₀) + 1, 3)
P = Vector{ComplexF64}
options = MonodromyOptions()
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

#### set up start pair
s0 = randn(ComplexF64, 2)
p0 = evaluate(r, s0)

### Monodromy
seed = rand(UInt32)
mon_result = monodromy_solve(
    MS,
    s0,
    p0,
    seed;
)

### parameters homotopy
start_parameters!(egtracker, p0)
target_parameters!(egtracker, zeros(2))
result = map(solutions(mon_result)) do s
    track(egtracker, s, 1.0)
end