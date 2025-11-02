using Oscar

# Incidence correspondance for discriminant
R, s, c, w = polynomial_ring(QQ, "s"=>1:2, "c"=>1:2, "w"=>1:2)

f1 = (s[1]*c[2]-c[1]*s[2]) + (s[1]*1 - c[1]*0) -3*w[1]
f2 = (s[2]*c[1]-c[2]*s[1]) + (s[2]*1 - c[2]*0) - 3*w[2]
g1 = s[1]^2 + c[1]^2 - 1
g2 = s[2]^2 + c[2]^2 - 1

Jac = derivative.( [f1, f2, g1, g2], transpose(vcat(s, c)) )
detJac = det(matrix(R, Jac))

F = [f1, f2, g1, g2, detJac]
I = ideal(F)

# Compute the discriminant
Ielim = eliminate(I, [s; c])
h = gens(Ielim)[1]