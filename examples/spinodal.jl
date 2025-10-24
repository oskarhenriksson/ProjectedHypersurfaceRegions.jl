# Example from thermodynamics
# This is a discriminant that controls the possible geometries that a spinodal curve can take
# in a phase diagram for a system with two species and no solvent-interactions

# Terminated within a couple of hours but gave incomplete solution set

using Pkg, Random
Pkg.activate(".")

include("../src/functions.jl");

Random.seed!(0x8b868320)

# The parameters of the model (interaction coefficients)
@var e11 e12 e22
ε = [e11, e12, e22]

# The variables of the model (concentrations of the two species)
@var x[1:2]

# Polynomial defining the spinodal curve
h = -4 * e11 * e22 * x[1]^2 * x[2] - 4 * e11 * e22 * x[1] * x[2]^2 + 
        4 * e12^2 * x[1]^2 * x[2] + 4 * e12^2 * x[1] * x[2]^2 + 
        4 * e11 * e22 * x[1] * x[2] - 4 * e12^2 * x[1] * x[2] + 
        2 * e11 * x[1]^2 + 4 * e12 * x[1] * x[2] + 
        2 * e22 * x[2]^2 - 2 * e11 * x[1] - 2 * e22 * x[2] + 1

# Set up incidence variety for spinodal discriminant
F = System([h, differentiate(h, x[1]), differentiate(h, x[2])], variables=[ε; x])

# Routing gradient
r = RoutingGradient(F, ε)

# This gives 6 but I know the discriminant has degree 72
d = degree(r.PWS) 

# Routing points
res, mon_res = critical_points(r)
pts = real_solutions(res)

# Connected components
G, idx, failed_info = partition_of_critical_points(r, pts)