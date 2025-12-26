using Random

include("../src/functions.jl");
include("./analysis.jl");
mkpath("./results/quadratic");

Random.seed!(0x8b868320)

# Set up the system
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])

# Pick a center for the routing function
c = [13, 2]

# Set up the routing function gradient
∇r = RoutingGradient(F, [a, b]; c=c)

d = degree(∇r.PWS)
println("Degree of discriminant: $d")

# Test forming the routing point homotopies
# p1 = zeros(2)
# q1 = randn(2)
# H = RoutingPointsHomotopy(r, p1, q1)
# u = randn(ComplexF64, 2)
# U = randn(ComplexF64, 2, 2)
# x0 = randn(ComplexF64, 2)
# t0 = 1.0
# @time evaluate_and_jacobian!(u, U, H, x0, t0)
# evaluate_and_jacobian!(u, U, H, x0, t0)

# Find the complex critical points 
# pts = read_solutions("./results/quadratic/routing_points.txt") |> real
pts, res, mon_res = critical_points(∇r)

write_parameters("./results/quadratic/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/quadratic/monodromy_result.txt", solutions(mon_res))
write_solutions("./results/quadratic/result.txt", solutions(res))
write_solutions("./results/quadratic/routing_points.txt", pts)

# Connect the critical points
G, idx, failed_info = partition_of_critical_points(∇r, pts)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failed_info)")
println()

write("./results/quadratic/connected_components.txt", string(G))

# Analyze root counts and plot result
M_x = maximum(p -> abs(p[1]), pts) + 4
M_y = maximum(p -> abs(p[2]), pts) + 3
analyze_result(∇r, pts, G, idx;
    h=(a, b) -> (a^2 - 4 * b),
    markersize=7,
    arrowstyle=:simple,
    flow_linewidth=3,
    discriminant_linewidth=4,
    root_counting_system=System([x^2 + a * x + b], variables=[x], parameters=[a; b]),
    legend=:bottomright,
    contour_stepsize=0.1,
    M_x_max=M_x,
    M_x_min=-M_x,
    M_y_max=M_y,
    M_y_min=-M_y,
)

savefig("./figures/quadratic.png")
savefig("./figures/quadratic.svg")
savefig("./figures/quadratic.pdf")