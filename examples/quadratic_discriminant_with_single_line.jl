using Random, Plots, DifferentialEquations, Random, Plots, DifferentialEquations, LightGraphs

include("../src/functions.jl");

Random.seed!(12345)

# Set up the system
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])

# Pick a center for the routing function
c = [10, 5]

# Set up the routing function gradient
∇r = RoutingGradient(F, [a, b]; c = c, g=[b])
e = denominator_exponent(∇r)

# Options for the monodromy step
options = MonodromyOptions(
    parameter_sampler = p -> 20 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 15 # change the stopping criterion
)

# Find the complex critical points 
pts, res0, mon_res = critical_points(∇r; options = options)

# Try another round of monodromy (only if you think the first attempt missed solutions)
#pts, res0, mon_res = critical_points(r, solutions(mon_res), parameters(mon_res), options = options)

# Connect the critical points
G, idx, failed_info = partition_of_critical_points(∇r, pts)
G

##### Plotting 
M_x_min = minimum(p -> p[1], pts) - 5
M_x_max = maximum(p -> p[1], pts) + 5
M_y_min = minimum(p -> p[2], pts) - 5
M_y_max = maximum(p -> p[2], pts) + 5

# Countour plot of the routing function
RR(x, y) = log(abs((x^2 - 4 * y) * y / (1 + (x-c[1])^2 + (y-c[2])^2)^e)) 
contour(
	M_x_min:0.1:M_x_max,
	M_y_min:0.1:M_y_max,
	RR,
	levels = 50,
	color = :plasma,
	clabels = false,
	cbar = false,
	lw = 1
);

# Plot the discriminant
A = [[b; b^2 / 4] for b in M_x_min:0.1:M_x_max]
plot!(
	Tuple.(A),
	xlims = (M_x_min, M_x_max),
	ylims = (M_y_min, M_y_max),
	linecolor = :black,
	linewidth = 4,
	label = "discriminant",
);

# Plot flows from the critical points
pts1 = pts[idx .!= 0]
g(x, param, t) = real(evaluate(∇r, x))
tspan = (0.0, 1e4)
for u0 in pts1
	jac = real(evaluate_and_jacobian(∇r, u0)[2])
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

savefig("./figures/quadratic_discriminant_with_single_line.png")
savefig("./figures/quadratic_discriminant_with_single_line.svg")
savefig("./figures/quadratic_discriminant_with_single_line.pdf")