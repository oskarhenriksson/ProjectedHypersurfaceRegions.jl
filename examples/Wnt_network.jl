## Method for applying functions to larger systems
# Want to show that Jacobian is singular without computing determinant
# Add non-zero dummy variable vector λ in the nullspace of the Jacobian
# To do this, add equations Jac*λ=0 and rand(19)'λ-1=0
# In addition to avoiding the slow and memory intensive determinant, this should be faster
# since it replaces a dense polynomial of high degree with 19 sparse polynomials of low degree


include("../src/functions.jl");

@var κ[1:31] x[1:19] T[1:5]
# Define the system of equations for the Wnt Signaling Pathway
# Goal: We want to say something about the real discriminant of this system (so eliminate the x's)
# k reaction rate coefficients
# c conservation constants
# x chemical concentrations

# Steady state equations (the commented out equations are redundant)
chemical_eqns = [
    #-κ[1]*x[1]+κ[2]*x[2], 
    κ[1] * x[1] - (κ[2] + κ[26]) * x[2] + κ[27] * x[3] - κ[3] * x[2] * x[4] +
    (κ[4] + κ[5]) * x[14],
    κ[26] * x[2] - κ[27] * x[3] - κ[14] * x[3] * x[6] + (κ[15] + κ[16]) * x[15],
    -κ[3] * x[2] * x[4] - κ[9] * x[4] * x[10] +
    κ[4] * x[14] +
    κ[8] * x[16] +
    (κ[10] + κ[11]) * x[18],
    -κ[28] * x[5] + κ[29] * x[7] - κ[6] * x[5] * x[8] + κ[5] * x[14] + κ[7] * x[16],
    -κ[14] * x[3] * x[6] - κ[20] * x[6] * x[11] +
    κ[15] * x[15] +
    κ[19] * x[17] +
    (κ[21] + κ[22]) * x[19],
    κ[28] * x[5] - κ[29] * x[7] - κ[17] * x[7] * x[9] + κ[16] * x[15] + κ[18] * x[17],
    -κ[6] * x[5] * x[8] + (κ[7] + κ[8]) * x[16],
    -κ[17] * x[7] * x[9] + (κ[18] + κ[19]) * x[17],
    κ[12] - (κ[13] + κ[30]) * x[10] - κ[9] * x[4] * x[10] + κ[31] * x[11] + κ[10] * x[18],
    -κ[23] * x[11] + κ[30] * x[10] - κ[31] * x[11] - κ[20] * x[6] * x[11] -
    κ[24] * x[11] * x[12] +
    κ[25] * x[13] +
    κ[21] * x[19],
    -κ[24] * x[11] * x[12] + κ[25] * x[13],
    κ[3] * x[2] * x[4] - (κ[4] + κ[5]) * x[14],
    κ[14] * x[3] * x[6] - (κ[15] + κ[16]) * x[15],
    κ[9] * x[4] * x[10] - (κ[10] + κ[11]) * x[18],
    #κ[20]*x[6]*x[11]-(κ[21]+κ[22])*x[19], 
    x[1] + x[2] + x[3] + x[14] + x[15] - T[1],
    x[4] + x[5] + x[6] + x[7] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] - T[2],
    x[8] + x[16] - T[3],
    x[9] + x[17] - T[4],
    x[12] + x[13] - T[5],
]

# Jacobian
Jac = differentiate(chemical_eqns,x)

#dJ = det(Jac) #This is taking forever!
# Because we can't take the determinant of the Jacobian we can still find when it is singular
# We use dummy variable λ and make it nonzero to give the Jacobian a non-trivial nullspace
# random λ constraint to enforece that λ nonzero
@var λ[1:19]
singular_jacobian = Jac*λ
λ_nonzero = rand(1:100, length(λ)) ⋅ λ - 1
#λ_nonzero = λ ⋅ λ - 1

# System for the incidence variety of the discriminant
F = System([chemical_eqns; singular_jacobian; λ_nonzero], variables = [κ; T; x; λ])

# Choose projection variables and reorder the system
all_vars = variables(F)
projection_vars = [κ; T]
x_vars = setdiff(all_vars, projection_vars)
F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
k = length(projection_vars)

# Computation of the pseudo witness set 
# Note: The initial witness set computation run out of memory regardless of start system  
PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1) 
PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1, start_system = :total_degree) 

e = floor(degree(PWS) / 2) + 1
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)
