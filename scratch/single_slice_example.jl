# An attempt to implement the single slice method for computing the Hessian

using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations, Plots

include("../src/functions.jl")

### a 2-dim example
@var x a b
F = System([x^2 + a * x + b; 2 * x + a])

projection_variables = [a; b]

c = 10 .* randn(2)
e = 2

# Fixed point where we will compute gradients and Hessians
P = [3; -2]

# For comparsion: the actual gradient and hessian of log(h)
h = a^2 - 4 * b
grad_symbolic = differentiate(log(h), [a; b])
grad_true = grad_symbolic([a; b] => P)
hess_symbolic = differentiate(grad_symbolic, [a; b])
hess_true = hess_symbolic([a; b] => P)

##################################

# single_slice method starts here
# Assumes that the variables of F are already ordered!

n = nvariables(F)
k = length(projection_variables)

# Random unit direction
# B = rand(k)
B = [2; 5]
B = B / norm(B);


# Evaluating the system on the line p+tβ 
@var u[1:n-k] p[1:k] β[1:k] t
F_on_line = F([p + t * β; u])

N = length(F_on_line)
@assert N == n-k+1 "Unexpected length of system"

# Symbolic Jacobians
JsuF = differentiate(F_on_line, vcat(t, u[:])) #Jacobian of F with respect to s and u
JPF = differentiate(F_on_line, p)
JBF = differentiate(F_on_line, β)

# Compute the intersection points through a pseudowitness set
# Note: The PWS should be given as an input of the function
# TODO: Use gradient cache for this 
# The tracking function would need to be adapted to the case of a single direction
PWS = PseudoWitnessSet(F, k; linear_subspace_codim=k - 1);
L_target = lifted_line(P, B, n)
list_of_solutions = solutions(HC.solve(PWS.F, PWS.W, start_subspace=PWS.L, target_subspace=L_target, intrinsic=true))
S = zeros(ComplexF64, length(list_of_solutions))
U = zeros(ComplexF64, n - k, length(list_of_solutions))
for (j, sol) in enumerate(list_of_solutions)
    X = sol[1:k]
    U[:, j] = sol[k+1:end]
    _, nonzero_coordinate = findmax(abs, X - P)
    S[j] = (X[nonzero_coordinate] - P[nonzero_coordinate]) / B[nonzero_coordinate]
end

# Compute first-order Jacobians of s and u
SP = zeros(ComplexF64, length(S), k)
SB = zeros(ComplexF64, length(S), k)
UP = zeros(ComplexF64, length(S), n - k, k)
UB = zeros(ComplexF64, length(S), n - k, k)
for i = 1:length(S)
    Jsu = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[i], U[:, i], P, B))
    JP = evaluate(JPF, vcat(t, u, p, β) => vcat(S[i], U[:, i], P, B))
    JB = evaluate(JBF, vcat(t, u, p, β) => vcat(S[i], U[:, i], P, B))
    A = -Jsu \ [JP JB] # solves the linear system Jsu * A = -[JP JB]
    SP[i, :] = A[1, 1:k]
    SB[i, :] = A[1, k+1:end]
    UP[i, :, :] = A[2:end, 1:k]
    UB[i, :, :] = A[2:end, k+1:end]
end



# Testing
hess_b_direction = sum(S[j]^(-2) * SP[j, :] for j = 1:length(S))
norm(hess_true*B - hess_b_direction) # close to zero

grad = sum(S[j]^(-2) * SB[j, :] for j = 1:length(S))
norm(grad_true - grad) #also nearly zero.

# Compute second-order Jacobians of s and use this to estimate the Hessian
A = zeros(ComplexF64, length(S), size(F_on_line, 1), k, k)
for i = 1:length(F_on_line)# loop through every equation in F?
    HF = differentiate(differentiate(F_on_line[i], vcat(t, u)), vcat(t, u))
    JxB = differentiate(differentiate(F_on_line[i], vcat(t, u)), β)
    JxP = differentiate(differentiate(F_on_line[i], vcat(t, u)), p)
    JPB = differentiate(differentiate(F_on_line[i], p), β)
    for j = 1:length(S)
        H = evaluate(HF, vcat(t, u, p, β) => vcat(S[j], U[:, j], P, B))
        Jxb = evaluate(JxB, vcat(t, u, p, β) => vcat(S[j], U[:, j], P, B))
        Jxp = evaluate(JxP, vcat(t, u, p, β) => vcat(S[j], U[:, j], P, B))
        Jpb = evaluate(JPB, vcat(t, u, p, β) => vcat(S[j], U[:, j], P, B))
        A[j, i, :, :] = ([SP[j, :] transpose(UP[j, :, :])] * H * transpose([SB[j, :] transpose(UB[j, :, :])])
                         + Jpb + [SP[j, :] transpose(UP[j, :, :])] * Jxb
                         + transpose(Jxp) * transpose([SB[j, :] transpose(UB[j, :, :])])) |> transpose |> Matrix

    end
end

hess1 = zeros(ComplexF64, k, k)
hess2 = zeros(ComplexF64, k, k)
for j = 1:length(S)
    Jtu = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[j], U[:, j], P, B))
    sols = zeros(ComplexF64, k, k)
    for a in 1:k, b in 1:k
        rhs = vcat([A[j, i, a, b] for i = 1:length(F_on_line)]...)
        sols[a, b] = -(Jtu\rhs)[1]
    end
    hess1 = hess1 - 2 * S[j]^(-3) * SP[j, :] * transpose(SB[j, :])
    hess2 = hess2 + S[j]^(-2) * sols
end

hess = hess1 + hess2 |> real
norm(hess_true - hess) # should be close to zero
