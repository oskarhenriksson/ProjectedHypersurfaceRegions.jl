using Oscar

# Set up the system
R, p, x = polynomial_ring(QQ, "p"=>1:2, "x"=>1:3)
a = p[1]
b = p[2]
steady_state = [
    x[1]*(1 - x[1])*(x[1] - b) - 2*a*x[1] + a*x[2] + a*x[3], 
    x[2]*(1 - x[2])*(x[2] - b) - 2*a*x[2] + a*x[1] + a*x[3], 
    x[3]*(1 - x[3])*(x[3] - b) - 2*a*x[3] + a*x[1] + a*x[2]
]

# Boundary discriminant 
# (detects parameters where we have zeros on the boundary other than (0,0,0))
I_steady_state = ideal(steady_state)
I_sat = saturation(I_steady_state, ideal(x))
I_boundary_incidence = I_sat + ideal([x[1]*x[2]*x[3]])
I_boundary_discriminant = eliminate(I_boundary_incidence, x)
h_boundary_discriminant = gens(I_boundary_discriminant)[1]

# Singularity discriminant
# (detects parameters where we have singular steady states)
# NOTE: Does not terminate within an hour!
Jac = derivative.(steady_state, transpose(x))
detJac = det(matrix(R, Jac))
F = [steady_state; detJac]
I_singular_incidence = ideal(F)
I_singular_discriminant = eliminate(I_singular_incidence, x)
h_singular_discriminant = gens(I_singular_discriminant)[1]


