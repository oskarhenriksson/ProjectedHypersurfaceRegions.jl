# Ecological system from https://www.worldscientific.com/doi/abs/10.1142/S0218127421502023
# Unknown computation time

using Pkg, Random, Plots, ImplicitPlots
Pkg.activate(".")

include("../src/functions.jl");

Random.seed!(0x8b868320)

### Set up incidence variety of discriminant
@var s[1:2] c[1:2] w[1:2]
freq1 = (s[1]*c[2]-c[1]*s[2]) + (s[1]*1 - c[1]*0) -3*w[1]
freq2 = (s[2]*c[1]-c[2]*s[1]) + (s[2]*1 - c[2]*0) - 3*w[2]
norm1 = s[1]^2 + c[1]^2 - 1
norm2 = s[2]^2 + c[2]^2 - 1
steady_state = [freq1, freq2, norm1, norm2]
Jac = differentiate.(steady_state, [s; c]')
detJac = expand(det(Jac)/4)

F = System([steady_state; detJac], variables = [s; c; w])

### Routing gradient
C = rand(2)
∇r = RoutingGradient(F, w, c=C);
d = degree(∇r.PWS)

### Critical points

# Find the complex critical points with monodromy
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 10 # change the stopping criterion
)

res, mon_res = critical_points(∇r, options = options)
write_parameters("kuramoto_monodromy_parameters.txt", parameters(mon_res))
write_solutions("kuramoto_monodromy_result.txt", solutions(mon_res)) 
write_solutions("kuramoto_result.txt", solutions(res))

# Try another round of monodromy (only if you think the first attempt missed solutions)
res, mon_res = critical_points(r, solutions(mon_res), parameters(mon_res), options = options)
write_parameters("kuramoto_monodromy_parameters.txt", parameters(mon_res))
write_solutions("kuramoto_monodromy_result.txt", solutions(mon_res)) 
write_solutions("kuramoto_result.txt", solutions(res))

# Extract the real critical points
pts = real_solutions(res)
write_solutions("kuramoto_routing_points.txt", pts)

### Connected components
G, idx, failed_info = partition_of_critical_points(∇r, pts)
write_solutions("kuramoto_components.txt", map(g -> Int.(g), G))
write_parameters("kuramoto_components.txt", Int.(idx))

### Plotting
M_x = maximum(p -> abs(p[1]), pts)*1.05
M_y = maximum(p -> abs(p[2]), pts)*1.05

# Plot the discriminant
h(x, y) = 314928*x^8*y^4 + 1259712*x^7*y^5 + 1889568*x^6*y^6 + 1259712*x^5*y^7 + 
    314928*x^4*y^8 + 139968*x^10 + 699840*x^9*y + 1277208*x^8*y^2 + 
    909792*x^7*y^3 - 279936*x^6*y^4 - 1084752*x^5*y^5 - 279936*x^4*y^6 + 
    909792*x^3*y^7 + 1277208*x^2*y^8 + 699840*x*y^9 + 
    139968*y^10 - 96957*x^8 - 387828*x^7*y - 226962*x^6*y^2 + 676512*x^5*y^3 + 
    1128249*x^4*y^4 + 676512*x^3*y^5 - 226962*x^2*y^6 - 387828*x*y^7 - 96957*y^8 + 
    22680*x^6 + 68040*x^5*y - 20844*x^4*y^2 - 155088*x^3*y^3 - 20844*x^2*y^4 + 
    68040*x*y^5 + 22680*y^6 - 2298*x^4 - 4596*x^3*y - 6894*x^2*y^2 - 4596*x*y^3 - 2298*y^4 + 
    96*x^2 + 96*x*y + 96*y^2 - 1;

implicit_plot(
    h; 
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 6,
    label = "discriminant",
	legend = false
)

# Plot the routing function
e = degree(∇r.PWS)
R(x, y) = log(abs(h(x,y) / (1 + (x-C[1])^2 + (y-C[2])^2)^e))
contour(
    (-M_x):0.01:M_x,
    (-M_y):0.01:M_y,
    R,
    levels = 50,
    color = :plasma,
    clabels = false,
    cbar = false,
    lw = 1,
    fill = true,
)

# Plot gradient flow
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
	plot!(flow[1:k], linecolor = :steelblue, linewidth = 2, label = false, arrow = true)
	plot!(flow[k:end], linecolor = :steelblue, linewidth = 2, label = false)
	prob = ODEProblem(g, u0 - 0.01*v, tspan)
	sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
	flow = Tuple.(sol.u)
	l = length(flow)
	k = div(l, 3)
	plot!(flow[1:k], linecolor = :steelblue, linewidth = 2, label = false, arrow = true)
	plot!(flow[k:end], linecolor = :steelblue, linewidth = 2, label = false)
end

# Plot the critical points
palette = collect(range(colorant"darkgreen", stop=colorant"lightgreen", length=length(G)))
for (i, component) in enumerate(G)
    scatter!(Tuple.(pts[component]), markercolor = palette[i], markersize = 3, label = "Critical points in region $i")
end

plot!(; legend = false, dpi=400, legendfontsize=6, yticks=false, xticks=false)

savefig("./figures/kuramoto.svg")
savefig("./figures/kuramoto.png")