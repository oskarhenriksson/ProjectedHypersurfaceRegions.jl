# An attempt to implement the single slice method for computing the Hessian

using HomotopyContinuation, LinearAlgebra, DifferentialEquations, Plots

include("../src/functions.jl")

# Example
@var p[1:2] x
F = System([x^2 + p[1] * x + p[2]; 2 * x + p[1]])

# Given point P
P = [3; -2]

# True gradient and Hessian
disc = p[1]^2 - 4 * p[2]
symbolic_grad = differentiate(log(disc), p)
actual_grad = symbolic_grad(p => P)
symbolic_hess = differentiate(symbolic_grad, p)
actual_hess = symbolic_hess(p => P)

# Reorder the system
all_vars = variables(F)
x = setdiff(all_vars, p)
F = System(F.expressions, variables=[p; x])

n = nvariables(F)
k = length(p)
N = n - k + 1 # number of equations in F
@assert N == length(F)

# Random unit vector for direction
# B = rand(k)
B = [2; 5]
B = B / norm(B)

# Form the pseudo witness set (TODO: Use gradient cache for this)
PWS = PseudoWitnessSet(F, k; linear_subspace_codim=k - 1);
L_target = lifted_line(P, B, n)
list_of_solutions = solutions(HC.solve(PWS.F, PWS.W, start_subspace=PWS.L, target_subspace=L_target, intrinsic=true))

# The system evaluated on the line v=α+sβ
@var α[1:k] β[1:k] s
v = α + s * β
F_on_line = F([v; x])
F = expressions(F)

# Symbolic derivatives
F_x = differentiate(F, x)(p=>v) 
F_p = differentiate(F, p)(p=>v)

# F_pp[i,j,ell] is ∂²F_i/∂p_j∂p_ell
F_pp = zeros(Expression, N, k, k)
for i = 1:N
    F_pp[i, :, :] = differentiate(differentiate(F[i], p), p)(p => v)
end

# F_px[i,j,ell] is ∂²F_i/∂p_j∂x_ell
F_px = zeros(Expression, N, k, n - k)
for i = 1:N
    F_px[i, :, :] = differentiate(differentiate(F[i], p), x)(p => v)
end

# F_xp[i,j,ell] is ∂²F_i/∂x_j∂p_ell
F_xp = zeros(Expression, N, n - k, k)
for i = 1:N
    F_xp[i, :, :] = differentiate(differentiate(F[i], x), p)(p => v)
    @assert F_xp[i, :, :] == transpose(F_px[i, :, :])
end

# F_xx[i,j,ell] is ∂²F_i/∂x_j∂x_ell
F_xx = zeros(Expression, N, n - k, n - k)
for i = 1:N
    F_xx[i, :, :] = differentiate(differentiate(F[i], x), x)(p => v)
end


grad_estimation = zeros(1, k)
Hess_estimation = zeros(k, k)
for sol in list_of_solutions
    #TODO: Skip solutions for which we have NaN (but warn the user about this)

    V = sol[1:k]
    X = sol[k+1:end]
    @assert length(X) == n - k
    _, nonzero_coordinate = findmax(abs, V - P)
    S = (V[nonzero_coordinate] - P[nonzero_coordinate]) / B[nonzero_coordinate]

    F_p_evaluated = F_p(α => P, β => B, s => S, x => X)
    F_x_evaluated = F_x(α => P, β => B, s => S, x => X)

    F_xx_evaluated = F_xx(α => P, β => B, s => S, x => X)
    F_xp_evaluated = F_xp(α => P, β => B, s => S, x => X)
    F_px_evaluated = F_px(α => P, β => B, s => S, x => X)
    F_pp_evaluated = F_pp(α => P, β => B, s => S, x => X)

    # Estimate the first order derivatives of S and X by solving a linear system
    # TODO: Since the coefficient matrix is the same for both the α and β deriviatives, 
    # we could merge to a single linear system
    coefficient_matrix = hcat(F_p_evaluated * B, F_x_evaluated)
    
    linsol = coefficient_matrix \ -F_p_evaluated    
    S_α = linsol[1:1, :]
    X_α = linsol[2:end, :]

    linsol = coefficient_matrix \ -(S .* F_p_evaluated)
    S_β = linsol[1:1, :]
    X_β = linsol[2:end, :]

    P_β = B * S_β + S * diagm(ones(k))

    # Estimate the gradient
    grad_estimation += S^(-2) * S_β
    
    # Now estimate the second-order derivatives of S and X by solving another linear system
    second_order_coefficient_matrix = zeros(ComplexF64, N * k, k + k * (n - k))
    right_hand_side = zeros(ComplexF64, N * k, k)
    for i = 1:N
        for j = 1:k
            second_order_coefficient_matrix[(i-1)*k+j, j] = transpose(B) * F_p_evaluated[i, :]
            second_order_coefficient_matrix[(i-1)*k+j, k+(j-1)*(n-k)+1:k+j*(n-k)] = F_x_evaluated[i, :]
        end
        right_hand_side[(i-1)*k+1:i*k, :] =
            F_pp_evaluated[i, :, :] * P_β + F_px_evaluated[i, :, :] * X_β
            + transpose(S_α) * (transpose(F_p_evaluated[i, :]) + transpose(B) * (F_pp_evaluated[i, :, :] * P_β + F_px_evaluated[i, :, :] * X_β))
            + transpose(X_α) * (F_xp_evaluated[i, :, :] * P_β + F_xx_evaluated[i, :, :] * X_β)
    end
    second_order_linsol = second_order_coefficient_matrix \ -right_hand_side
    @assert second_order_coefficient_matrix * second_order_linsol ≈ -right_hand_side
    S_αβ = second_order_linsol[1:k, 1:k]
    Hess_estimation += -2 * S^(-3) * transpose(S_β) * S_α + S^(-2) * transpose(S_αβ)
end

# Outputs to return
real(grad_estimation)
@assert norm(actual_grad' - real(grad_estimation)) < 1e-12
real(Hess_estimation)
real(Hess_estimation) - actual_hess
