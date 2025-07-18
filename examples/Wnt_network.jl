## Method for applying functions to larger systems
# Want to show that Jacobian is singular without computing determinant
# Add non-zero dummy variable vector l in the nullspace of the Jacobian
# To do this, add equations Jac*l=0 and rand(19)'l-1=0
# In addition to avoiding the slow and memory intensive determinant, this should be faster
# since it replaces a dense polynomial of high degree with 19 spare polynomials of low degree

using Pkg
Pkg.activate("..")
include("../src/functions.jl");


@var κ[1:31] x[1:19] T[1:5] l[1:19]
# Define the system of equations for the Wnt Signaling Pathway
# Goal: We want to say something about the real discriminant of this system (so eliminate the x's)
# k reaction rate coefficients
# c conservation constants
# x chemical concentrations
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
#The commented out equations are redundant.

Jac = differentiate(chemical_eqns,x)
#dJ = det(Jac) #This is taking forever!
# Because we can't take the determinant of the Jacobian we can still find when it is singular
# We use dummy variable l and make it nonzero to give the Jacobian a non-trivial nullspace
lconst = l[1]+l[2]+3*l[3]+l[4]+l[5]+l[6]+l[7]+l[8]+l[9]+3*l[10]+l[11]+l[12]+l[13]+l[14]+l[15]+l[16]+3*l[17]+l[18]+l[19]-1 
# random l constraint to make l nonzero
F = System(vcat(chemical_eqns,Jac*l,lconst), variables = [κ; T; x; l])
all_vars = variables(F)
projection_vars = [κ; T]
x_vars = setdiff(all_vars, projection_vars)
F_ordered = System(F.expressions, variables = [projection_vars; x_vars])

# System for the incidence variety of the discriminant
k = length(projection_vars)
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)
e = 20
# e should be at least  floor(degree(PWS)/2) + 1


PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1) # issue! Our linear space intersects at something larger than points.

# u + tv this in R^(36). Lift it to R^(55). So codimension of (u+tv) was 35, and the pullback to R^(55) still has codimension 35. So it has dimension 20.
L = create_line(u, v, size(F_ordered, 2))
#WITHIN PseudoWitnessSet
n = ambient_dim(L)
startL = rand_subspace(n; codim = codim(L))
ourWitnessSet = witness_set(F, startL) # Get an error here.
