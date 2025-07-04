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
e = denominator_exponent(r)

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

### parameter homotopy
start_parameters!(egtracker, p0)
target_parameters!(egtracker, zeros(2))
result = HomotopyContinuation.solve(H, solutions(mon_result))
pts = real_solutions(result)




##### Plotting 
g(x, param, t) = real(evaluate(r, x))
u0 = randn(2)
tspan = (0.0, 1e4)
prob = ODEProblem(g, u0, tspan)
sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)

M = maximum(abs, vcat(pts...)) + 2
M_x = maximum(p -> abs(p[1]), pts) + 2
M_y = maximum(p -> abs(p[2]), pts) + 2

R(x, y) = log(abs((x^2 - 4 * y) / (1 + (x-c[1])^2 + (y-c[2])^2)^e))
contour(
    (-M_x):0.1:M_x,
    (-M_y):0.1:M_y,
    R,
    levels = 50,
    color = :plasma,
    clabels = false,
    cbar = false,
    lw = 1,
    fill = true,
)
A = [[b; b^2 / 4] for b = (-M_x):M_x]
plot!(
    Tuple.(A),
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 8,
    label = "discriminant",
)

scatter!(Tuple.(pts), markercolor = :green, markersize = 8, label = "critical points")
plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label = "gradient flow")
#scatter!([Tuple(u0)], markercolor=:blue, markersize=8, label="gradient flow start")

plot!(; legend = false)