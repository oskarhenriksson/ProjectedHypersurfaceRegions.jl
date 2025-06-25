# This is our attempt to compute the Wint chemical reaction network discriminant.
# Cant continue till we find the determinant of the jacobian, which is a 19x19 matrix. )

using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations

include("../src/functions.jl");


@var k[1:31] x[1:19] c[1:5]
# Define the system of equations for the Wnt Signaling Pathway
# Goal: We want to say something about the real discriminant of this system (so eliminate the x's)
# k reaction rate coefficients
# c conservation constants
# x chemical concentrations
chemical_eqns = [
    #-k[1]*x[1]+k[2]*x[2], 
    k[1]*x[1] - (k[2]+k[26])*x[2]+k[27]*x[3]-k[3]*x[2]*x[4]+(k[4]+k[5])*x[14],
    k[26]*x[2]-k[27]*x[3]-k[14]*x[3]*x[6]+(k[15]+k[16])*x[15],
    -k[3]*x[2]*x[4]-k[9]*x[4]*x[10]+k[4]*x[14]+k[8]*x[16]+(k[10]+k[11])*x[18],
    -k[28]*x[5]+k[29]*x[7]-k[6]*x[5]*x[8]+k[5]*x[14]+k[7]*x[16],
    -k[14]*x[3]*x[6]-k[20]*x[6]*x[11]+k[15]*x[15]+k[19]*x[17]+(k[21]+k[22])*x[19],
    k[28]*x[5]-k[29]*x[7]-k[17]*x[7]*x[9]+k[16]*x[15]+k[18]*x[17],
    -k[6]*x[5]*x[8]+(k[7]+k[8])*x[16], 
    -k[17]*x[7]*x[9]+(k[18]+k[19])*x[17], 
    k[12]-(k[13]+k[30])*x[10]-k[9]*x[4]*x[10]+k[31]*x[11]+k[10]*x[18],
    -k[23]*x[11]+k[30]*x[10]-k[31]*x[11]-k[20]*x[6]*x[11]-k[24]*x[11]*x[12]+k[25]*x[13]+k[21]*x[19], 
    -k[24]*x[11]*x[12]+k[25]*x[13],
    k[3]*x[2]*x[4]-(k[4]+k[5])*x[14], 
    k[14]*x[3]*x[6]-(k[15]+k[16])*x[15], 
    k[9]*x[4]*x[10]-(k[10]+k[11])*x[18], 
    #k[20]*x[6]*x[11]-(k[21]+k[22])*x[19], 
    x[1]+x[2]+x[3]+x[14]+x[15]-c[1], 
    x[4]+x[5]+x[6]+x[7]+x[14]+x[15]+x[16]+x[17]+x[18]+x[19]-c[2],
    x[8]+x[16]-c[3],
    x[9]+x[17]-c[4], 
    x[12]+x[13]-c[5]]
#The commented out equations are redundant.

#Jac = differentiate(chemical_eqns,x)
#dJ = det(Jac) #This is taking forever!
# So lets just try to eliminats the x's from the chemical_eqns (without the Jacobian).
F = System(chemical_eqns, variables = [k; c; x])
# System for the incidence variety of the discriminant
myk = 36
B = qr(rand(myk,myk)).Q |> Matrix
myc = 10 .* randn(myk)
e = 20
# e should be at least  floor(degree(PWS)/2) + 1
all_vars = variables(F)
projection_vars = [k;c]
x_vars = setdiff(all_vars, projection_vars)
F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
PWS = PseudoWitnessSet(F_ordered, myk, myk-1) # issue! Our linear space intersects at something larger than points.

# u + tv this in R^(36). Lift it to R^(55). So codimension of (u+tv) was 35, and the pullback to R^(55) still has codimension 35. So it has dimension 20.
L = create_line(u, v, size(F_ordered, 2))
#WITHIN PseudoWitnessSet
n = ambient_dim(L)
startL = rand_subspace(n; codim = codim(L))
ourWitnessSet = witness_set(F, startL) # Get an error here.