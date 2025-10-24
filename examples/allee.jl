# Ecological system from https://www.worldscientific.com/doi/abs/10.1142/S0218127421502023
# Unknown computation time

using Pkg, Random
Pkg.activate(".")

include("../src/functions.jl");

Random.seed!(0x8b868320)

### Set up incidence variety of discriminant
@var a b x[1:3]
steady_state = [x[1]*(1 - x[1])*(x[1] - b) - 2*a*x[1] + a*x[2] + a*x[3], 
x[2]*(1 - x[2])*(x[2] - b) - 2*a*x[2] + a*x[1] + a*x[3], 
x[3]*(1 - x[3])*(x[3] - b) - 2*a*x[3] + a*x[1] + a*x[2]]
Jac = differentiate.(steady_state, x')
detJac = det(Jac)
F = System([steady_state; x[1]*x[2]*x[3]*detJac], variables = [a; b; x])

### Routing gradient
r = RoutingGradient(F, [a,b]);
d = degree(r.PWS) 

### Routing points
res, mon_res = critical_points(r)
pts = real_solutions(res)

### Connected components
G, idx, failed_info = partition_of_critical_points(r, pts)