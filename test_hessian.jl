using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations

include("src/functions.jl");


@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])
# System for the incidence variety of the discriminant

B = qr(rand(2, 2)).Q |> Matrix
c = 10 .* randn(2)
e = 2
k = 2

###### Critical points 
#pts = include("discr_pw.jl")

hess_off_diag = hess_log_r(F, e, k; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F, e, k; method = :many_slices, c, B)

# You can also call hess_log_r(F, e, k; method = :off_diag, c, B)
# c and B are optional -- otherwise, they are taken randomly.

# if you just pass in the discriminant, we can compute the Hessian directly. 
# (So this is definitely correct)
disc = a^2 - 4 * b
actual_hess = hess_log_r_given_h(disc, e; c)


p = rand(2)
@time actual_hess(p) # 119 allocations
@time hess_off_diag(p) # 671 allocations
@time hess_many_slices(p) # 2.46 k allocations
u, U = randn(Float64, k), randn(Float64, k, k)
U

# This does not work?
# r = RoutingGradient(F, [a; b]; c = c, B = B)
# @time evaluate_and_jacobian!(u, U, r, p)
# @time _evaluate_jacobian(r,p)


@time for _ = 1:1000
    p = rand(2)
    hess_off_diag(p)
end

@time for _ = 1:1000
    p = rand(2)
    hess_many_slices(p)
end

## A 3-dimension discriminant example

@var α β γ x
F = System([x^3 + α * x^2 + β * x + γ; 3 * x^2 + 2 * α * x + β], variables = [α, β, γ, x])
# System for the incidence variety of the discriminant

B = qr(rand(3, 3)).Q |> Matrix
c = 10 .* randn(3)
e = 4
k = 3


hess_off_diag = hess_log_r(F, e, k; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F, e, k; method = :many_slices, c, B)

disc = α^2 * β^2 - 4 * β^3 - 4 * α^3 * γ - 27 * γ^2 + 18 * α * β * γ
actual_hess = hess_log_r_given_h(disc, e; c)

# These are all the same!
p = rand(3)
hess_off_diag(p)
hess_many_slices(p)
actual_hess(p)

# Time analysis
@time for _ = 1:1000
    p = rand(3)
    hess_off_diag(p)
end
@time for _ = 1:1000
    p = rand(3)
    hess_many_slices(p)
end


# Here is a general example for the discriminant of a polynomial of degree 'degf' in 'x'.
degf = 6
@var a[0:degf] x
f = sum([a[i+1] * x^i for i = 0:degf])
df = differentiate(f, x)
F = System([f; df], variables = [a; x])


k = degf + 1
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)
e = 4

hess_off_diag = hess_log_r(F, e, k; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F, e, k; method = :many_slices, c, B)

P = rand(k)
hess_off_diag(P)
hess_many_slices(P)

# This may slow down a lot for deg f big -- starts to take forever at degf = 20. Not sure of a faster way.
m = 2degf - 1
f_rows = [vcat(zeros(i), a, zeros(m - degf - i - 1)) for i = 0:m-degf-1]
da = [i * a[i+1] for i = 0:degf][2:end]
df_rows = [vcat(zeros(i), da, zeros(m - degf - i)) for i = 0:m-degf]
sylvester = stack(vcat(f_rows, df_rows))
discf = det(sylvester) / a[degf+1]
actual_hess = hess_log_r_given_h(discf, e; c)
actual_hess(P)
