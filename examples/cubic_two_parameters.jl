
using ImplicitPlots, Plots, Random

include("../src/functions.jl");

Random.seed!(0x8b868320)

########

# The discriminant of x^3 + a * x^2 + b*x + 1 is
# 4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27
@var a b x
F = System([x^3 + a * x^2 + b * x + 1; 3 * x^2 + 2 * a * x + b], variables=[a, b, x])

projection_variables = [a; b]
k = length(projection_variables)

# Routing gradient
c = [13.758979284873828, -0.09884333335596635]
∇r = RoutingGradient(F, projection_variables; c=c)

# Degree of discriminant
d = degree(∇r.PWS)
println("Degree of discriminant: $d")

# Critical points
pts, res0, mon_res = critical_points(∇r)

# Connecting 
G, idx, failed_info = partition_of_critical_points(∇r, pts)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failed_info)")
println()

# Analyzw result
include("./analysis.jl");
M_x = maximum(p -> abs(p[1]), pts) + 6
M_y = maximum(p -> abs(p[2]), pts) + 6
h(x, y) = 4 * x^3 - x^2 * y^2 - 18 * x * y + 4 * y^3 + 27
analyze_result(∇r, pts, G, idx;
    h=h,
    markersize=7,
    arrowstyle=:simple,
    flow_linewidth=3,
    discriminant_linewidth=4,
    legend=:bottomright,
    root_counting_system=System([x^3 + a * x^2 + b * x + 1], variables=[x], parameters=[a; b]),
    M_x_max=M_x,
    M_x_min=-M_x,
    M_y_max=M_y,
    M_y_min=-M_y,
)

savefig("./figures/cubic.pdf")
savefig("./figures/cubic.svg")
savefig("./figures/cubic.png")