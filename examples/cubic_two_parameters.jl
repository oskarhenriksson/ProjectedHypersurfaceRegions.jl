using Pkg
Pkg.activate(".")

using ImplicitPlots, Plots, DifferentialEquations, Random

include("../src/functions.jl");

Random.seed!(0x8b868320)

########

# The discriminant of x^3 + a * x^2 + b*x + 1 is
# 4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27
@var a b x
F = System([x^3 + a * x^2 + b*x + 1; 3*x^2 + 2*a*x + b], variables = [a, b, x])

projection_variables = [a; b]
k = length(projection_variables)

B = qr(rand(k, k)).Q |> Matrix # not needed anymore (but kept for reproducibility)
c = 10 .* randn(k)
r = RoutingGradient(F, projection_variables; c = c)
e = denominator_exponent(r)

p1 = zeros(2)
q1 = randn(2)
H = RoutingPointsHomotopy(r, p1, q1)

### Use monodromy to the system ∇r = p0 where we view the right-hand side are the parameters of the system
egtracker = EndgameTracker(H) # we want to add options later 
trackers = [egtracker]
x₀ = zeros(ComplexF64, size(H, 2))

unique_points = UniquePoints(
    x₀,
    1;
)

trace = zeros(ComplexF64, length(x₀) + 1, 3)
P = Vector{ComplexF64}
options = MonodromyOptions(
    #parameter_sampler = p -> 10 .* [0; randn(ComplexF64, length(p) - 1)], # bigger lopps
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger lopps
    # max_loops_no_progress = 10 # change the stopping criterion
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

#### set up start pair
s0 = randn(ComplexF64, k)
p0 = evaluate(r, s0)


u = randn(ComplexF64, k)
U = randn(ComplexF64, k, k)
@time evaluate_and_jacobian!(u, U, r, s0)

using Profile, PProf
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 evaluate_and_jacobian!(u, U, r, s0)   
PProf.Allocs.pprof()


### Monodromy 
seed = rand(UInt32)
mon_result = monodromy_solve(
    MS,
    s0,
    p0,
    seed;
)

### Use a parameter homotopy (using the "detour trick") to trace the solutions of ∇r = p0 to solutions of ∇r = 0
start_parameters!(egtracker, p0)
p_intermediate = randn(ComplexF64, 2)
target_parameters!(egtracker, p_intermediate)
intermediate_result = HomotopyContinuation.solve(H, solutions(mon_result))


start_parameters!(egtracker, p_intermediate)
target_parameters!(egtracker, zeros(2))
result = HomotopyContinuation.solve(H, solutions(intermediate_result))

pts = real_solutions(result)

##### Plotting 
M_x = maximum(p -> abs(p[1]), pts) + 6
M_y = maximum(p -> abs(p[2]), pts) + 6


# Discriminant
h(x, y) = 4*x^3 - x^2*y^2 - 18*x*y + 4*y^3 + 27

R(x, y) = log(abs(h(x,y) / (1 + (x-c[1])^2 + (y-c[2])^2)^e))
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


implicit_plot!(
    h; 
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 6,
    label = "discriminant"
)

# Gradient flow for routing points of positive index
g(x, param, t) = real(evaluate(r, x))
tspan = (0.0, 1e4)
for (i, u0) in enumerate(pts)
    println()
    println("Critical point #$i")
    jac = real(evaluate_and_jacobian(r, u0)[2])
    eigen_data = eigen(jac)
    eigenvalues = eigen_data.values
    eigenvectors = eigen_data.vectors
    positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
    println("Index: $(length(positive_directions))")
    if !isempty(positive_directions)
        idx = first(positive_directions)
        v = eigenvectors[:, idx]
        prob = ODEProblem(g, u0 + 0.01*v, tspan)
        sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
        plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label="")
        prob = ODEProblem(g, u0 - 0.01*v, tspan)
        sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
        plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label="")
        scatter!(Tuple(u0), markercolor = :magenta, markersize = 8, label = "")
    else
        scatter!(Tuple(u0), markercolor = :green, markersize = 8, label = "")
    end

end

# Legend
plot!([], [], color = :steelblue, linewidth = 4, label = "gradient flow")
scatter!(Tuple(NaN), markercolor = :green, markersize = 8, label = "routing pts (index 0)")
scatter!(Tuple(NaN), markercolor = :magenta, markersize = 8, label = "routing pts (index > 0)")

plot!(; legend = :bottomright, dpi=400, legendfontsize=6)

savefig("./figures/example_cubic.svg")
savefig("./figures/example_cubic.png")