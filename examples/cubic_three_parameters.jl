using Random

include("../src/functions.jl");

Random.seed!(0x8b868320)

########

# The discriminant of x^3 + a * x^2 + b*x + γ is
# 4*a^3*γ - a^2*b^2 - 18*a*b*γ + 4*b^3 + 27*γ^2
@var a b γ x
f = x^3 + a * x^2 + b * x + γ
F = System([f; differentiate(f, x)], variables=[a, b, γ, x])
projection_variables = [a; b; γ]
k = length(projection_variables)

# Routing gradient
c = 10 .* randn(k)
∇r = RoutingGradient(F, projection_variables; c=c)

# Degree of discriminant
d = degree(∇r.PWS)
println("Degree of discriminant: $d")

# critical points
pts, res, mon_res = critical_points(∇r)

### connecting 
G, idx, failed_info = partition_of_critical_points(∇r, pts)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failed_info)")
println()

analyze_result(∇r, pts, G, idx;
    root_counting_system=System([x^3 + a * x^2 + b * x + γ], variables=[x], parameters=[a; b; γ])
)