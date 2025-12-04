using Random, Plots, DifferentialEquations, Random, Plots, DifferentialEquations, LightGraphs

include("../src/functions.jl");

Random.seed!(12345)

# Set up the system
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])

# Pick a center for the routing function
c = [10, 5]

# Set up the routing function gradient
#r = RoutingGradient(F, [a, b]; c = c);
r = RoutingGradient(F, [a, b]; c = c, g=[a, b]);
e = denominator_exponent(r)

# Test evaluation
r([1,4])
u = randn(ComplexF64, 2)
U = randn(ComplexF64, 2, 2)
x0 = [1,4] #randn(ComplexF64, 2)
@time evaluate_and_jacobian!(u, U, r, x0)
u
U

# Options for the monodromy step
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 10 # change the stopping criterion
)

# Find the complex critical points 
pts, res0, mon_res = critical_points(r; start_grid_width=0, options = options)
#pts, res0, mon_res = critical_points(r; options = options) # fails at gradient flow step

# Try another round of monodromy (only if you think the first attempt missed solutions)
#pts, res0, mon_res = critical_points(r, solutions(mon_res), parameters(mon_res), options = options)

# Connect the critical points
G, idx, failed_info = partition_of_critical_points(r, pts)
G


for pt in pts
	u, U = evaluate_and_jacobian(r, pt)
	eigenvalues = LinearAlgebra.eigen(U).values
	println("Point: $pt")
	println("Eigenvalues: $(real.(eigenvalues))")
end

##### Plotting 
M_x = maximum(p -> abs(p[1]), pts) + 10
M_y = maximum(p -> abs(p[2]), pts) + 10

# Countour plot of the routing function
RR(x, y) = log(abs((x^2 - 4 * y) * x * y / (1 + (x-c[1])^2 + (y-c[2])^2)^e)) 
contour(
	(-M_x):0.1:M_x,
	(-M_y):0.1:M_y,
	RR,
	levels = 50,
	color = :plasma,
	clabels = false,
	cbar = false,
	lw = 1
);

# Plot the discriminant
A = [[b; b^2 / 4] for b in (-M_x):M_x]
plot!(
	Tuple.(A),
	xlims = (-M_x, M_x),
	ylims = (-M_y, M_y),
	linecolor = :black,
	linewidth = 4,
	label = "discriminant",
);

# Plot flows from the critical points
pts1 = pts[idx .!= 0]
g(x, param, t) = real(evaluate(r, x))
tspan = (0.0, 1e4)
for u0 in pts1
	jac = real(evaluate_and_jacobian(r, u0)[2])
	eigen_data = LinearAlgebra.eigen(jac)
	eigenvalues = eigen_data.values
	eigenvectors = eigen_data.vectors
	positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
	j = first(positive_directions)
	v = eigenvectors[:, j]

	prob = ODEProblem(g, u0 + 0.01*v, tspan)
	sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
	flow = Tuple.(sol.u)
	l = length(flow)
	k = div(l, 3)
	plot!(flow[1:k], linecolor = :steelblue, linewidth = 3, label = false, arrow = true)
	plot!(flow[k:end], linecolor = :steelblue, linewidth = 3, label = false)
	prob = ODEProblem(g, u0 - 0.01*v, tspan)
	sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
	flow = Tuple.(sol.u)
	l = length(flow)
	k = div(l, 3)
	plot!(flow[1:k], linecolor = :steelblue, linewidth = 3, label = false, arrow = true)
	plot!(flow[k:end], linecolor = :steelblue, linewidth = 3, label = false)
end

# Plot the critical points (colored by component)
palette = collect(range(colorant"green", stop=colorant"lightgreen", length=length(G)));
for (i, component) in enumerate(G)
    scatter!(Tuple.(pts[component]), markercolor = palette[i], markersize = 8, label = "Critical points in region $i")
end;

plot!(; legend = :bottomright, dpi=400, legendfontsize=6)

savefig("./figures/quadratic_discriminant_with_lines.png")
savefig("./figures/quadratic_discriminant_with_lines.svg")
savefig("./figures/quadratic_discriminant_with_lines.pdf")