using Pkg, Random
Pkg.activate(".")
include("../src/functions.jl");

Random.seed!(0x8b868320)

#example 5.2 in the overleaf file

a2 = 1.4; a3 = 0.7; b3 = 1.0; A2 = 1.6; A3 = 0.9; B3 = 0.6;

@var p[1:2] φ[1:2] c[1:2]

f = [ φ[1]^2 + φ[2]^2 - 1,
      p[1]^2 + p[2]^2 - 2*(a3*p[1] +b3*p[2])*φ[1] + 2*(b3*p[1] - a3*p[2])*φ[2] + a3^2 + b3^2 - c[1],
      p[1]^2 + p[2]^2 - 2*A2*p[1] + 2*((a2-a3)*p[1] - b3*p[2] + A2*a3 -A2*a2)*φ[1] + 2*(b3*p[1] + (a2-a3)*p[2] - A2*b3)*φ[2]
        + (a2-a3)^2 + b3^2 + A2^2 -c[2],
      p[1]^2 + p[2]^2 -2*(A3*p[1] + B3*p[2]) + A3^2 + B3^2 - 1.0
    ]

J = differentiate(f, vcat(p,φ))
D = det(J)

#Since we want the numerator of the routing function to have D(c)*c, 
#we add D*c to the system before we project down. Not sure about this. 
F = System([f; D])
c
projection_variables = c

k = length(projection_variables)

γ = 10 .* randn(k)
r = RoutingGradient(F, projection_variables; c = γ)

# critical points
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 10 # change the stopping criterion
)
res = critical_points(r, options = options)
pts = real_solutions(res)

# connecting 
G, idx, failed_info = partition_of_critical_points(r, pts)
G