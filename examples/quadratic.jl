using Pkg, Random, Plots, DifferentialEquations, Random, Plots, DifferentialEquations, LightGraphs
Pkg.activate(".")

include("../src/functions.jl");

Random.seed!(0x8b868320)

########
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])

c = [13.758979284873828, -0.09884333335596635]
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


### monodromy
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 10 # change the stopping criterion
)


res0 = critical_points(r, options = options)
pts = real_solutions(res0)

### connecting 
G, idx, failed_info = partition_of_critical_points(r, pts)
G



##### Plotting 
M_x = maximum(p -> abs(p[1]), pts) + 4
M_y = maximum(p -> abs(p[2]), pts) + 3


RR(x, y) = log(abs((x^2 - 4 * y) / (1 + (x-c[1])^2 + (y-c[2])^2)^2)) #This is our routing function
contour(
	(-M_x):0.1:M_x,
	(-M_y):0.1:M_y,
	RR,
	levels = 40,
	color = :plasma,
	clabels = false,
	cbar = false,
	lw = 1,
	fill = true,
)

A = [[b; b^2 / 4] for b in (-M_x):M_x] #discriminant of the quadratic #discriminant of the quadratic
plot!(
	Tuple.(A),
	xlims = (-M_x, M_x),
	ylims = (-M_y, M_y),
	linecolor = :black,
	linewidth = 8,
	label = "discriminant",
)


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

plot!(; legend = :bottomright, dpi=400, legendfontsize=6)

# # Gradient flow for routing points of positive index
# g(x, param, t) = real(evaluate(r, x))
# tspan = (0.0, 1e4)
# for (i, u0) in enumerate(pts)
#     println()
#     println(i)
#     jac = real(evaluate_and_jacobian(r, u0)[2])
#     eigen_data = eigen(jac)
#     eigenvalues = eigen_data.values
#     eigenvectors = eigen_data.vectors
#     positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
#     println("Index: $(length(positive_directions))")
#     if !isempty(positive_directions)
#         idx = first(positive_directions)
#         v = eigenvectors[:, idx]
#         prob = ODEProblem(g, u0 + 0.01*v, tspan)
#         sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
#         plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label="")
#         prob = ODEProblem(g, u0 - 0.01*v, tspan)
#         sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
#         plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label="")
#         scatter!(Tuple(u0), markercolor = :magenta, markersize = 8, label = "")
#     else
#         scatter!(Tuple(u0), markercolor = :green, markersize = 8, label = "")
#     end

# end

# # Legend
# plot!([], [], color = :steelblue, linewidth = 4, label = "gradient flow")
# scatter!(Tuple(NaN), markercolor = :green, markersize = 8, label = "routing pts (index 0)")
# scatter!(Tuple(NaN), markercolor = :magenta, markersize = 8, label = "routing pts (index > 0)")

# plot!(; legend = :bottomright, dpi=400, legendfontsize=6)

# savefig("./figures/example_quadratic.png")
# savefig("./figures/example_quadratic.svg")
