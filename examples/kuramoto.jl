# Algebraic formulation of the Kuramoto model with three oscillators
# See, e.g., https://doi.org/10.1063/1.4919696

using Pkg
Pkg.activate(".")
using Random, Plots, ImplicitPlots

include("../src/functions.jl");

Random.seed!(12345)

time_start = time()

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
write_parameters("./results/kuramoto/center.txt", C)
∇r = RoutingGradient(F, w, c=C);
d = degree(∇r.PWS)

### Critical points

# Find the complex critical points with monodromy
options = MonodromyOptions(
    parameter_sampler = p -> 20 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 15 # change the stopping criterion
)

res, mon_res = critical_points(∇r, options = options)

write_parameters("./results/kuramoto/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/kuramoto/monodromy_result.txt", solutions(mon_res)) 
write_solutions("./results/kuramoto/result.txt", solutions(res))

# Extract the real critical points
pts = real_solutions(res)
write_solutions("./results/kuramoto/routing_points.txt", pts)

### Connected components
G, idx, failed_info = partition_of_critical_points(∇r, pts)
write_solutions("./results/kuramoto/components.txt", map(g -> Int.(g), G))
write_parameters("./results/kuramoto/components.txt", Int.(idx))

time_end = time()
println("Computation time: $(time_end - time_start) seconds")

### Plotting
M_x = maximum(p -> abs(p[1]), pts)*1.05
M_y = maximum(p -> abs(p[2]), pts)*1.05

h(x, y) = 314928*x^8*y^4 + 1259712*x^7*y^5 + 1889568*x^6*y^6 + 1259712*x^5*y^7 + 
    314928*x^4*y^8 + 139968*x^10 + 699840*x^9*y + 1277208*x^8*y^2 + 
    909792*x^7*y^3 - 279936*x^6*y^4 - 1084752*x^5*y^5 - 279936*x^4*y^6 + 
    909792*x^3*y^7 + 1277208*x^2*y^8 + 699840*x*y^9 + 
    139968*y^10 - 96957*x^8 - 387828*x^7*y - 226962*x^6*y^2 + 676512*x^5*y^3 + 
    1128249*x^4*y^4 + 676512*x^3*y^5 - 226962*x^2*y^6 - 387828*x*y^7 - 96957*y^8 + 
    22680*x^6 + 68040*x^5*y - 20844*x^4*y^2 - 155088*x^3*y^3 - 20844*x^2*y^4 + 
    68040*x*y^5 + 22680*y^6 - 2298*x^4 - 4596*x^3*y - 6894*x^2*y^2 - 4596*x*y^3 - 2298*y^4 + 
    96*x^2 + 96*x*y + 96*y^2 - 1;

# Plot the routing function
e = ∇r.e
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

# Plot the discriminant
implicit_plot!(
    h; 
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 2,
    label = "discriminant",
	legend = false
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
	plot!(flow[1:k], linecolor = :steelblue, linewidth = 2, label = false, arrow = :closed)
	plot!(flow[k:end], linecolor = :steelblue, linewidth = 2, label = false)
	prob = ODEProblem(g, u0 - 0.01*v, tspan)
	sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
	flow = Tuple.(sol.u)
	l = length(flow)
	k = div(l, 3)
	plot!(flow[1:k], linecolor = :steelblue, linewidth = 2, label = false, arrow = :closed)
	plot!(flow[k:end], linecolor = :steelblue, linewidth = 2, label = false)
end

# Plot the critical points
palette = collect(range(colorant"darkgreen", stop=colorant"lightgreen", length=length(G)))
for (i, component) in enumerate(G)
    scatter!(Tuple.(pts[component]), markercolor = palette[i], markersize = 3, label = "Critical points in region $i")
end

plot!(; legend = true, dpi=400, legendfontsize=6)
#plot!(; legend = false, dpi=400, legendfontsize=6, yticks=false, xticks=false)

savefig("./figures/kuramoto.svg")
savefig("./figures/kuramoto.png")

plot!(, xlims = (-1, 1), ylims = (-1, 1))

savefig("./figures/kuramoto_zoomed_in.svg")
savefig("./figures/kuramoto_zoomed_in.png")


# Try another round of monodromy (only if you think the first attempt missed solutions)
old_number_of_monodromy_solutions = length(solutions(mon_res))
options = MonodromyOptions(
    parameter_sampler = p -> 100 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 15 # change the stopping criterion
)
res, mon_res = critical_points(∇r, solutions(mon_res), parameters(mon_res), options = options)
if length(solutions(mon_res)) > old_number_of_monodromy_solutions
    println("Found new solutions with additional monodromy round!")
    write_parameters("./results/kuramoto/monodromy_parameters.txt", parameters(mon_res))
    write_solutions("./results/kuramoto/monodromy_result.txt", solutions(mon_res)) 
    write_solutions("./results/kuramoto/result.txt", solutions(res))
else
    println("No new solutions found in the additional monodromy round.")
end

# If new solutions were found, repeat the steps above manually!