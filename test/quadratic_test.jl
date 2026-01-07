using Pkg
include("../src/functions.jl");
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])



c = randn(2)
# Set up the routing function gradient
R = RoutingGradient(F, [a, b]; c = c);



@var a, b
e = denominator_exponent(R)
disc = a^2 - 4 * b
q = 1 + sum(([a;b] - c).^2)
r_test = differentiate(log(disc / q^e), [a;b])
H_test = differentiate(r_test, [a;b])

k = 2
p = randn(ComplexF64, k)
u1 = evaluate(r_test, [a;b] => p)
U1 = evaluate(H_test, [a;b] => p)
u2, u22, U2 = randn(ComplexF64, k), randn(ComplexF64, k), randn(ComplexF64, k, k);
@time evaluate_and_jacobian!(u2, U2, R, p);
@time evaluate!(u22, R, p);
u2
U2

norm(u1 - u2)
norm(u1 - u22)
norm(U1 - U2)