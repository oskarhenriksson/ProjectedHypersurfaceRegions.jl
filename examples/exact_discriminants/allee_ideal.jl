using Oscar

# Incidence correspondance for discriminant
R, p, x = polynomial_ring(QQ, "p"=>1:2, "x"=>1:3)
a = p[1]
b = p[2]
steady_state = [
    x[1]*(1 - x[1])*(x[1] - b) - 2*a*x[1] + a*x[2] + a*x[3], 
    x[2]*(1 - x[2])*(x[2] - b) - 2*a*x[2] + a*x[1] + a*x[3], 
    x[3]*(1 - x[3])*(x[3] - b) - 2*a*x[3] + a*x[1] + a*x[2]
]
Jac = derivative.(steady_state, transpose(x))
detJac = det(matrix(R, Jac))
F = [steady_state; detJac]
I = ideal(F)

# Attempt: Guess of the discriminant (from https://arxiv.org/pdf/2501.19062)
# h_guess = 2*a*b*(-1//2 + b)*(-1//2*a^2*b^12 - 475//8*a^2*b^6 + 27//4*a*b^8 + 339//2*a^3*b^4 - 45//8*a*b^9 + 287//4*a^2*b^7 + 51//16*a*b^10 - 475//8*a^2*b^8 - 9//8*a*b^11 + 265//8*a^2*b^9 + 3//16*a*b^12 - 97//8*a^2*b^10 + 339//2*a^3*b^8 - 1//2*a^2*b^2 + 4*a^4 + 856*a^6 + 34992*a^9 - 3888*a^8 - 2592*a^7 - 96*a^5 - 1//64*b^12 + 3//32*b^11 - 15//64*b^10 + 5//16*b^9 - 15//64*b^8 + 3//32*b^7 - 1//64*b^6 - 9//8*a*b^5 + 3//16*a*b^4 - 69*a^4*b^2 - 570*a^5*b^3 + 1086*a^4*b^5 - 258*a^5*b^2 - 852*a^4*b^4 + 396*a^4*b^7 - 97//8*a^2*b^4 - 60*a^3*b^9 + 3*a^2*b^11 + 3246*a^6*b^2 + 984*a^5*b^4 - 852*a^4*b^6 - 2568*a^6*b + 5184*a^7*b^3 - 2568*a^6*b^5 + 384*a^5*b^7 - 20*a^4*b^9 - 7776*a^7*b^2 + 3246*a^6*b^4 - 258*a^5*b^6 - 69*a^4*b^8 + 12*a^3*b^10 + 5184*a^7*b - 2212*a^6*b^3 - 570*a^5*b^5 - 3888*a^8*b^2 - 2592*a^7*b^4 + 856*a^6*b^6 - 96*a^5*b^8 + 4*a^4*b^10 + 3888*a^8*b - 20*a^4*b - 60*a^3*b^3 + 12*a^3*b^2 + 3*a^2*b^3 + 384*a^5*b + 396*a^4*b^3 - 318*a^3*b^7 + 393*a^3*b^6 - 318*a^3*b^5 + 51//16*a*b^6 + 265//8*a^2*b^5 - 45//8*a*b^7)
# h_guess in I # returns false

# Compute the discriminant
# NOTE: Does not terminate within an hour!
Ielim = eliminate(I, x)
h = gens(Ielim)[1]
