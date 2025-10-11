using Pkg, Random, Plots, DifferentialEquations, Random, Plots, DifferentialEquations
Pkg.activate(".")

include("src/functions.jl");

Random.seed!(0x8b868320)

########
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])

B = qr(rand(2, 2)).Q |> Matrix # not needed anymore (but kept for reproducibility)
c = randn(2)
r = RoutingGradient(F, [a, b]; c = c)

p1 = zeros(2)
q1 = randn(2)
H = RoutingPointsHomotopy(r, p1, q1)

### Test evaluation
u = randn(ComplexF64, 2)
U = randn(ComplexF64, 2, 2)
x0 = randn(ComplexF64, 2)
t0 = 1.0
@time evaluate_and_jacobian!(u, U, H, x0, t0)
evaluate_and_jacobian!(u, U, H, x0, t0)









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
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger lopps
    max_loops_no_progress = 20 # change the stopping criterion
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
s0 = randn(ComplexF64, 2)
p0 = evaluate(r, s0)

evaluate(r, s0, p0) #should give zero

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
M_x = maximum(p -> abs(p[1]), pts) + 4
M_y = maximum(p -> abs(p[2]), pts) + 3

#c = 2 .* randn(2)

R(x, y) = log(abs((x^2 - 4 * y) / (1 + (x-c[1])^2 + (y-c[2])^2)^2)) #This is our routing function
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


A = [[b; b^2 / 4] for b = (-M_x):M_x] #discriminant of the quadratic #discriminant of the quadratic
plot!(
    Tuple.(A),
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 8,
    label = "discriminant",
)

# Gradient flow for routing points of positive index
g(x, param, t) = real(evaluate(r, x))
tspan = (0.0, 1e4)
for (i, u0) in enumerate(pts)
    println()
    println(i)
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

savefig("figures/example_quadratic.png")
savefig("figures/example_quadratic.svg")