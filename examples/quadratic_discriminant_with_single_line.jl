using Random

include("../src/functions.jl");

Random.seed!(12345)

# Set up the system
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])

# Pick a center for the routing function
c = [10, 5]

# Set up the routing function gradient
∇r = RoutingGradient(F, [a, b]; c=c, g=[b])
e = denominator_exponent(∇r)

# Find the complex critical points 
pts, res0, mon_res = critical_points(∇r)

# Connect the critical points
G, idx, failed_info = partition_of_critical_points(∇r, pts)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failed_info)")
println()

##### Plotting 
include("./analysis.jl");
M_x_min = minimum(p -> p[1], pts) - 5
M_x_max = maximum(p -> p[1], pts) + 5
M_y_min = minimum(p -> p[2], pts) - 5
M_y_max = maximum(p -> p[2], pts) + 5
analyze_result(∇r, pts, G, idx;
    h=(a, b) -> (a^2 - 4 * b) * (b),
    markersize=7,
    arrowstyle=:simple,
    flow_linewidth=3,
    discriminant_linewidth=4,
    legend=:bottomright,
    contour_stepsize=0.1,
    M_x_max=M_x_max,
    M_x_min=M_x_min,
    M_y_max=M_y_max,
    M_y_min=M_y_min,
)
savefig("./figures/quadratic_discriminant_with_single_line.png")
savefig("./figures/quadratic_discriminant_with_single_line.svg")
savefig("./figures/quadratic_discriminant_with_single_line.pdf")