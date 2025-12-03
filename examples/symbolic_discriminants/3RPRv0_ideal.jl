using Oscar

# Set up incidence variety of the discriminant
a2 = 14//10; a3 = 7//10; b3 = 1; A2 = 16//10; A3 = 9//10; B3 = 6//10; c3 = 1;
R, φ, p, c = polynomial_ring(QQ, "φ"=>1:2, "p"=>1:2, "c"=>1:2)

f = [ φ[1]^2 + φ[2]^2 - 1,
      p[1]^2 + p[2]^2 - 2*(a3*p[1] +b3*p[2])*φ[1] + 2*(b3*p[1] - a3*p[2])*φ[2] + a3^2 + b3^2 - c[1],
      p[1]^2 + p[2]^2 - 2*A2*p[1] + 2*((a2-a3)*p[1] - b3*p[2] + A2*a3 -A2*a2)*φ[1] + 2*(b3*p[1] + (a2-a3)*p[2] - A2*b3)*φ[2]
        + (a2-a3)^2 + b3^2 + A2^2 -c[2],
      p[1]^2 + p[2]^2 -2*(A3*p[1] + B3*p[2]) + A3^2 + B3^2 - c3
    ]

Jac = derivative.(f, transpose(vcat(p,φ)))
detJac = det(matrix(R, Jac))

I = ideal(R, [f; detJac])

# Compute the discriminant
Ielim = eliminate(I, vcat(φ, p))
h = gens(Ielim)[1]

