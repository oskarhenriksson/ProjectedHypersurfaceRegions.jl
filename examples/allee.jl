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
F = System([steady_state; detJac], variables = [a; b; x])

# Routing gradient
C = randn(2)
write_parameters("./results/allee/center.txt", C)

# C = [-0.3597289068234817, 1.0872084924285859]

∇r = RoutingGradient(F, [a, b], c=C, g=[a, b, b+1, 3*a+b]);
d = degree(∇r.PWS)

# Routing points
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
)
pts, res, mon_res = critical_points(∇r, options = options;           
                            start_grid_width = 3,
                            start_grid_stepsize = 0.1,
                            start_grid_center = [1.51;1.51])

# pts = [[-1.2753320611557284, 1.162287272706235], [1.5691451025738021, 1.2422665498540606], [0.3828861403174712, 2.253913786718969], [-0.3425656704455543, -0.04855293339761104], [-2.4228839270962306, -7.031421185836445], [-3.817452310614039, -1.6616584699084351], [-0.4340831018742778, -0.7870931656784308], [0.03609728338434495, 0.4150454793987127], [4.5188684960064975, -1.9277678140449355], [-0.14154619678442495, 0.5421316611432299], [-0.6325207980893147, 2.154754780101732], [-0.49008272909465556, 1.5052369685844034], [0.12269239571653136, 1.4764323107853656], [-5.40177515696785, -4.69084889402527], [0.42439718340159654, 2.3618344262873827], [1.7474011219859231, 7.295662298539107], [8.630624909006785, -8.350042047309422], [17.08951703360676, 23.50191139800713], [-1.5583430297850598, 2.8399968166526204], [12.089362527752126, -41.46384269853712]]

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
is_nonnegative_solution = s -> all(xi -> (xi > 0 || abs(xi) < 1e-16), s)
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
e = denominator_exponent(∇r)
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