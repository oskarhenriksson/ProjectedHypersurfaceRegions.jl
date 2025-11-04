using Pkg
Pkg.activate(".")

using ImplicitPlots, Plots, Random

include("../src/functions.jl");

Random.seed!(0x8b868320)

########

# The discriminant of x^3 + a * x^2 + b*x + 1 is
# 4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27
@var a b x
F = System([x^3 + a * x^2 + b*x + 1; 3*x^2 + 2*a*x + b], variables = [a, b, x])

projection_variables = [a; b]
k = length(projection_variables)

c = [13.758979284873828, -0.09884333335596635]
r = RoutingGradient(F, projection_variables; c = c)

# Critical points
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 10 # change the stopping criterion
)

pts, res0, mon_res = critical_points(r, options = options)

# Connecting 
G, idx, failed_info = partition_of_critical_points(r, pts)
G

# Plotting 
M_x = maximum(p -> abs(p[1]), pts) + 6
M_y = maximum(p -> abs(p[2]), pts) + 6

# Discriminant
h(x, y) = 4*x^3 - x^2*y^2 - 18*x*y + 4*y^3 + 27
e = r.e
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


implicit_plot(
    h; 
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 6,
    label = "discriminant",
	legend = false
)
savefig("./figures/cubic_discriminant.png")


## plot flow
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

## plot critical points
palette = collect(range(colorant"darkgreen", stop=colorant"lightgreen", length=length(G)))
for (i, component) in enumerate(G)
    scatter!(Tuple.(pts[component]), markercolor = palette[i], markersize = 8, label = "Critical points in region $i")
end

plot!(; legend = false, dpi=400, legendfontsize=6, yticks=false, xticks=false)

savefig("./figures/example_cubic_without_legend.svg")
savefig("./figures/example_cubic.png")