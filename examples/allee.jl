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
steady_state = [
    x[1]*(1 - x[1])*(x[1] - b) - 2*a*x[1] + a*x[2] + a*x[3], 
    x[2]*(1 - x[2])*(x[2] - b) - 2*a*x[2] + a*x[1] + a*x[3], 
    x[3]*(1 - x[3])*(x[3] - b) - 2*a*x[3] + a*x[1] + a*x[2]
]
# steady_state = subs(steady_state, a => a / 10) # to make the scale in figure 3 in https://www.biorxiv.org/content/10.1101/2021.02.03.429609v2.full.pdf uniform
Jac = differentiate.(steady_state, x')
detJac = det(Jac)
F = System([steady_state; detJac], variables = [a; b; x])

# Routing gradient
C = randn(2)
write_parameters("./results/allee/center.txt", C)

∇r = RoutingGradient(F, [a, b], c=C, g=[a, b, b+1, 3*a+b]);

# Degree of discriminant
d = degree(∇r.PWS)
println("Degree of (singularity part of) discriminant: $d")

# Routing points
pts, res, mon_res = critical_points(∇r;           
                            start_grid_width = 3,
                            start_grid_stepsize = 0.1,
                            start_grid_center = [1.51;1.51])

# Connected components
G, idx, failed_info = partition_of_critical_points(∇r, pts)

# Record computation time
t_end_round1 = time()
println("Computation time for round 1: $(t_end_round1 - t_start_round1) seconds")

# Analyze result
include("./analysis.jl");
h = (a, b) -> (-1/2*a^2*b^12 - 475/8*a^2*b^6 + 27/4*a*b^8 + 339/2*a^3*b^4 - 45/8*a*b^9 + 287/4*a^2*b^7 + 51/16*a*b^10 - 475/8*a^2*b^8 - 9/8*a*b^11 + 265/8*a^2*b^9 + 3/16*a*b^12 - 97/8*a^2*b^10 + 339/2*a^3*b^8 - 1/2*a^2*b^2 + 4*a^4 + 856*a^6 + 34992*a^9 - 3888*a^8 - 2592*a^7 - 96*a^5 - 1/64*b^12 + 3/32*b^11 - 15/64*b^10 + 5/16*b^9 - 15/64*b^8 + 3/32*b^7 - 1/64*b^6 - 9/8*a*b^5 + 3/16*a*b^4 - 69*a^4*b^2 - 570*a^5*b^3 + 1086*a^4*b^5 - 258*a^5*b^2 - 852*a^4*b^4 + 396*a^4*b^7 - 97/8*a^2*b^4 - 60*a^3*b^9 + 3*a^2*b^11 + 3246*a^6*b^2 + 984*a^5*b^4 - 852*a^4*b^6 - 2568*a^6*b + 5184*a^7*b^3 - 2568*a^6*b^5 + 384*a^5*b^7 - 20*a^4*b^9 - 7776*a^7*b^2 + 3246*a^6*b^4 - 258*a^5*b^6 - 69*a^4*b^8 + 12*a^3*b^10 + 5184*a^7*b - 2212*a^6*b^3 - 570*a^5*b^5 - 3888*a^8*b^2 - 2592*a^7*b^4 + 856*a^6*b^6 - 96*a^5*b^8 + 4*a^4*b^10 + 3888*a^8*b - 20*a^4*b - 60*a^3*b^3 + 12*a^3*b^2 + 3*a^2*b^3 + 384*a^5*b + 396*a^4*b^3 - 318*a^3*b^7 + 393*a^3*b^6 - 318*a^3*b^5 + 51/16*a*b^6 + 265/8*a^2*b^5 - 45/8*a*b^7)*(3*a + b)*(3*a - b + 1)*(b^2 + 3*a - b)*(4*a*b^4 - 36*a^2*b^2 - 8*a*b^3 - b^4 + 108*a^3 + 36*a^2*b + 12*a*b^2 + 2*b^3 - 36*a^2 - 8*a*b - b^2 + 4*a)*(b + 1);
root_counting_system=System(steady_state, variables = x, parameters = [a; b]);
is_nonnegative_solution = s -> all(xi -> (xi > 0 || abs(xi) < 1e-16), s)
function analyze_and_save_result()
    write_parameters("./results/allee/monodromy_parameters.txt", parameters(mon_res))
    write_solutions("./results/allee/monodromy_result.txt", solutions(mon_res)) 
    write_solutions("./results/allee/result.txt", solutions(res))
    write_solutions("./results/allee/routing_points.txt", pts)
    write("./results/allee/connected_components.txt", string(G))

    println("Connected components: $(G)")
    println("Indicies: $(idx)")
    println("Failed info: $(failed_info)")
    println()
    
    analyze_result(∇r, pts, G, idx;
        h = h,
        root_counting_system = root_counting_system,
        root_count_condition = is_nonnegative_solution,
        plot_contour=false
    )
    savefig("./figures/allee.pdf")
    savefig("./figures/allee.svg")
    savefig("./figures/allee.png")
    
    plot!(; xlims = (-1,M_x_max), ylims = (-1,M_y_max), legend=:topleft)
    savefig("./figures/allee.pdf")
    savefig("./figures/allee.svg")
    savefig("./figures/allee.png")
end
analyze_and_save_result()

# Try another round of monodromy (only if you think the first attempt missed solutions)
println("Running second round of monodromy...")
t_start_round2 = time()
old_number_of_monodromy_solutions = length(solutions(mon_res))
options = MonodromyOptions(
    parameter_sampler = p -> 100 .* randn(ComplexF64, length(p)), # smaller loops
    max_loops_no_progress = 15 # change the stopping criterion
)
pts, res, mon_res = critical_points(∇r, solutions(mon_res), parameters(mon_res), options = options)
G, idx, failed_info = partition_of_critical_points(∇r, pts)

t_end_round2 = time()
println("Additional computation time for round 2: $(t_end_round2 - t_start_round2) seconds")
println("Total computation time for round 1 and 2: $((t_end_round2 - t_start_round2) + (t_end_round1 - t_start_round1)) seconds")

if length(solutions(mon_res)) > old_number_of_monodromy_solutions
    println("Found new solutions with additional monodromy round!")
    analyze_and_save_result()
else
    println("No new solutions found in the additional monodromy round.")
end