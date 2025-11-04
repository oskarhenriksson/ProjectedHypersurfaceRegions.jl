# Ecological system from https://www.worldscientific.com/doi/abs/10.1142/S0218127421502023
# See also https://www.biorxiv.org/content/10.1101/2021.02.03.429609v2.full

using Pkg, Random
Pkg.activate(".")
include("../src/functions.jl");
mkpath("./results/allee")

Random.seed!(12345)

t_start = time()

# Set up incidence variety of discriminant
@var a b x[1:3]
steady_state = [
    x[1]*(1 - x[1])*(x[1] - b) - 2*a*x[1] + a*x[2] + a*x[3], 
    x[2]*(1 - x[2])*(x[2] - b) - 2*a*x[2] + a*x[1] + a*x[3], 
    x[3]*(1 - x[3])*(x[3] - b) - 2*a*x[3] + a*x[1] + a*x[2]
]
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
    max_loops_no_progress = 15 # change the stopping criterion
)
pts, res, mon_res = critical_points(∇r, options = options)

write_parameters("./results/allee/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/allee/monodromy_result.txt", solutions(mon_res)) 
write_solutions("./results/allee/result.txt", solutions(res))
write_solutions("./results/allee/routing_points.txt", pts)

# Connected components
G, idx, failed_info = partition_of_critical_points(∇r, pts)
write("./results/allee/connected_components.txt", string(G))

t_end = time()
println("Computation time: $(t_end - t_start) seconds")

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

# Try another round of monodromy (only if you think the first attempt missed solutions)
println("Running second round of monodromy...")
old_number_of_monodromy_solutions = length(solutions(mon_res))
options = MonodromyOptions(
    parameter_sampler = p -> 100 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 15 # change the stopping criterion
)
res, mon_res = critical_points(∇r, solutions(mon_res), parameters(mon_res), options = options)
if length(solutions(mon_res)) > old_number_of_monodromy_solutions
    println("Found new solutions with additional monodromy round!")
else
    println("No new solutions found in the additional monodromy round.")
end

# If new solutions were found, repeat the steps above manually!