# Ecological system from https://www.worldscientific.com/doi/abs/10.1142/S0218127421502023
# See also https://www.biorxiv.org/content/10.1101/2021.02.03.429609v2.full
# More recent paper: https://arxiv.org/pdf/2501.19062

using Random, Plots, ImplicitPlots
include("../src/functions.jl");
mkpath("./results/allee")

Random.seed!(1234)

t_start_round1 = time()

# Set up incidence variety of discriminant
@var a b x[1:3]
steady_state1 = [
    x[1]*(1 - x[1])*(x[1] - b) - 2*a*x[1] + a*x[2] + a*x[3], 
    x[2]*(1 - x[2])*(x[2] - b) - 2*a*x[2] + a*x[1] + a*x[3], 
    x[3]*(1 - x[3])*(x[3] - b) - 2*a*x[3] + a*x[1] + a*x[2]
]
steady_state = subs(steady_state1, a => a / 10) # to make the scale in figure 3 in https://www.biorxiv.org/content/10.1101/2021.02.03.429609v2.full.pdf uniform
Jac = differentiate.(steady_state, x')
detJac = det(Jac)
F = System([steady_state; x[1]*x[2]*x[3]*detJac], variables = [a; b; x])

# Routing gradient
C = randn(2)
write_parameters("./results/allee/center.txt", C)
∇r = RoutingGradient(F, [a,b], c=C);
d = degree(∇r.PWS)

# Routing points
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
)
pts, res, mon_res = critical_points(∇r, options = options;           
                            start_grid_width = 3,
                            start_grid_stepsize = 0.1,
                            start_grid_center = [1.51;1.51])

# pts = [[1.5008505041416076, -4.992054374172119], [8.873775251675713, 10.184539724387184], [21.352372122219965, 26.73062909286873], [-9.516787924356157, 32.510119627058735], [10.144425852538403, 35.413484056491725], [0.22748939289880724, -2.9163837667277086], [2.479841961837091, -1.2421350166611869], [-0.142094211295228, 0.490144149893062], [-0.5548476367836471, 1.6960880661969613], [2.4025298982729666, -0.4765324163490884], [0.07023920300282123, -0.8760732585762937], [0.22143827315231507, 4.821953936949241], [0.09692923712483603, 4.814679303770955], [0.27409106612436296, -0.8832016985602994], [0.09848416350011838, -2.9980288906885226], [0.035627837070082796, 0.39605675328604417], [0.01975229042725789, 0.49002885154919423], [0.07615249752961624, 0.39138790391723843], [0.07583668038353164, 0.4889420759562086], [0.0521892333980436, 0.23706051426698405], [0.07530216907621935, 0.6210485701172596], [0.11807542738005739, 1.8172629904397462], [0.03630757960677417, 0.5962043763951855], [0.054330283033610524, 0.7449326022366476]]


write_parameters("./results/allee/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/allee/monodromy_result.txt", solutions(mon_res)) 
write_solutions("./results/allee/result.txt", solutions(res))
write_solutions("./results/allee/routing_points.txt", pts)

# Connected components
G, idx, failed_info = partition_of_critical_points(∇r, pts)
write("./results/allee/connected_components.txt", string(G))

t_end_round1 = time()
println("Computation time for round 1: $(t_end_round1 - t_start_round1) seconds")

# Analyze root counts
SS = System(steady_state, variables = x, parameters = [a; b])
is_nonnegative_solution = s -> all(xi -> (xi > 0 || xi ≈ 0), s)
for (i, comp) in enumerate(G)
    println("Connected component #$i")
    positive_root_counts = Int[]
    for j in comp
        real_steady_states = HC.solve(SS, target_parameters=pts[j]) |> real_solutions
        prc = filter(is_nonnegative_solution, real_steady_states) |> length
        push!(positive_root_counts, prc)
    end
    println("Positive root counts: $(positive_root_counts)\n")
end

# Plot result
# Plotting
M_x_max = maximum(p -> p[1], pts)*1.2
M_x_min = minimum(p -> p[1], pts)*1.2
M_y_max = maximum(p -> p[2], pts)*1.2
M_y_min = minimum(p -> p[2], pts)*1.2

# Guess of the discriminant (based on https://arxiv.org/pdf/2501.19062)
h(a, b) = (-1/2*a^2*b^12 - 475/8*a^2*b^6 + 27/4*a*b^8 + 339/2*a^3*b^4 - 45/8*a*b^9 + 287/4*a^2*b^7 + 51/16*a*b^10 - 475/8*a^2*b^8 - 9/8*a*b^11 + 265/8*a^2*b^9 + 3/16*a*b^12 - 97/8*a^2*b^10 + 339/2*a^3*b^8 - 1/2*a^2*b^2 + 4*a^4 + 856*a^6 + 34992*a^9 - 3888*a^8 - 2592*a^7 - 96*a^5 - 1/64*b^12 + 3/32*b^11 - 15/64*b^10 + 5/16*b^9 - 15/64*b^8 + 3/32*b^7 - 1/64*b^6 - 9/8*a*b^5 + 3/16*a*b^4 - 69*a^4*b^2 - 570*a^5*b^3 + 1086*a^4*b^5 - 258*a^5*b^2 - 852*a^4*b^4 + 396*a^4*b^7 - 97/8*a^2*b^4 - 60*a^3*b^9 + 3*a^2*b^11 + 3246*a^6*b^2 + 984*a^5*b^4 - 852*a^4*b^6 - 2568*a^6*b + 5184*a^7*b^3 - 2568*a^6*b^5 + 384*a^5*b^7 - 20*a^4*b^9 - 7776*a^7*b^2 + 3246*a^6*b^4 - 258*a^5*b^6 - 69*a^4*b^8 + 12*a^3*b^10 + 5184*a^7*b - 2212*a^6*b^3 - 570*a^5*b^5 - 3888*a^8*b^2 - 2592*a^7*b^4 + 856*a^6*b^6 - 96*a^5*b^8 + 4*a^4*b^10 + 3888*a^8*b - 20*a^4*b - 60*a^3*b^3 + 12*a^3*b^2 + 3*a^2*b^3 + 384*a^5*b + 396*a^4*b^3 - 318*a^3*b^7 + 393*a^3*b^6 - 318*a^3*b^5 + 51/16*a*b^6 + 265/8*a^2*b^5 - 45/8*a*b^7)*(3*a + b)*(3*a - b + 1)*(b^2 + 3*a - b)*(4*a*b^4 - 36*a^2*b^2 - 8*a*b^3 - b^4 + 108*a^3 + 36*a^2*b + 12*a*b^2 + 2*b^3 - 36*a^2 - 8*a*b - b^2 + 4*a)*(b + 1)
e = ∇r.e
R(x, y) = log(abs(h(x,y) / (1 + (x-center[1])^2 + (y-center[2])^2)^e))


# contour(
#     (M_x_min):0.01:M_x_max,
#     (M_y_min):0.01:M_y_max,
#     R,
#     levels = 50,
#     color = :plasma,
#     clabels = false,
#     cbar = false,
#     lw = 1,
#     fill = true,
# )

implicit_plot(
    h; 
    xlims = (M_x_min, M_x_max),
    ylims = (M_y_min, M_y_max),
    linecolor = :black,
    linewidth = 2,
    label = "discriminant",
	legend = false,
    resolution = 1000,
    aspect_ratio = :auto,
)

palette = collect(range(colorant"darkgreen", stop=colorant"lightgreen", length=length(G)));
for (i, component) in enumerate(G)
    scatter!(Tuple.(pts[component]), markercolor = palette[i], markersize = 3, label = "Critical points in region $i")
end

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


plot!(; legend = false, dpi=400, legendfontsize=6)

savefig("./figures/allee.pdf")
savefig("./figures/allee.svg")
savefig("./figures/allee.png")

plot!(; xlims = (-1,M_x_max), ylims = (-1,M_y_max), legend=:topleft)

savefig("./figures/allee.pdf")
savefig("./figures/allee.svg")
savefig("./figures/allee.png")

# Try another round of monodromy (only if you think the first attempt missed solutions)
println("Running second round of monodromy...")
t_start_round2 = time()
old_number_of_monodromy_solutions = length(solutions(mon_res))
options = MonodromyOptions(
    parameter_sampler = p -> 100 .* randn(ComplexF64, length(p)), # smaller loops
    max_loops_no_progress = 15 # change the stopping criterion
)
pts, res, mon_res = critical_points(∇r, solutions(mon_res), parameters(mon_res), options = options)
if length(solutions(mon_res)) > old_number_of_monodromy_solutions
    println("Found new solutions with additional monodromy round!")
    write_parameters("./results/allee/monodromy_parameters.txt", parameters(mon_res))
    write_solutions("./results/allee/monodromy_result.txt", solutions(mon_res)) 
    write_solutions("./results/allee/result.txt", solutions(res))
    write_solutions("./results/allee/routing_points.txt", pts)
    G, idx, failed_info = partition_of_critical_points(∇r, pts)
    write("./results/allee/connected_components.txt", string(G))
else
    println("No new solutions found in the additional monodromy round.")
end

t_end_round2 = time()
println("Additional computation time for round 2: $(t_end_round2 - t_start_round2) seconds")
println("Total computation time for round 1 and 2: $((t_end_round2 - t_start_round2) + (t_end_round1 - t_start_round1)) seconds")

# If new solutions were found, repeat the steps above manually!