# An attempt to implement the single slice method for computing the Hessian

using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, DifferentialEquations, Plots

include("../src/functions.jl")

### a 2-dim example
@var x a b
F = System([x^2 + a*x + b; 2*x + a])

projection_variables = [a; b]

c = 10 .* randn(2)
e = 2

# single_slice method starts here

n = nvariables(F)
k = length(projection_variables)

# random unit vector for direction
B = [2; 5] 
B = B/norm(B) 

#B = Matrix(qr(randn(k, k)).Q)

@var u[1:n-k] p[1:k] β[1:k] t

# evaluating the system on the line p+tβ 
F_on_line = F([p + t * β; u])

# getting the jacobians
JsuF = differentiate(F_on_line, vcat(t, u[:])) #Jacobian of F with respect to s and u

JPF = differentiate(F_on_line, p)

JBF = differentiate(F_on_line, β)

#set up function for hess(q)
q = 1 + sum((p - c) .* (p - c))
Hlogqe(pt) = evaluate(differentiate(differentiate(log(q^e), p), p), p => pt) 


P = [3; -2] # random point

# Solve F_on_line for given point P and direction B
evaluated_F = System(subs(F_on_line, vcat(p,β)=> vcat(P, B)), variables = [t; u])
solution_result = HC.solve(evaluated_F)
list_of_solutions = solutions(solution_result; only_nonsingular = true)

# Store the s and u values in separate matrices 
S = zeros(ComplexF64, length(list_of_solutions))
U = zeros(ComplexF64, n-k, length(list_of_solutions))
for (j,sol) in enumerate(list_of_solutions)
    #display(sol)
    S[j] = sol[1]
    U[:, j] = sol[2:end] 
end 


SP = zeros(ComplexF64, length(S), k)
SB = zeros(ComplexF64, length(S), k)
UP = zeros(ComplexF64, length(S), k)
UB = zeros(ComplexF64, length(S), k)
for i = 1:length(S)
    Jsu = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[i], U[:,i], P, B))
    JP = evaluate(JPF, vcat(t, u, p, β) => vcat(S[i], U[:,i], P, B))
    JB = evaluate(JBF, vcat(t, u, p, β) => vcat(S[i], U[:,i], P, B))
    A = -Jsu \ [JP JB] # solves the linear system Jsu * A = -[JP JB]
    SP[i,:] = A[1, 1:k]
    SB[i,:] = A[1, k+1:end]
    UP[i,:] = A[2, 1:k]
    UB[i,:] = A[2, k+1:end]
end

#test 
h = a^2-4*b
Gh = evaluate(differentiate(log(h), [a;b]), [a; b] => P )
Hh = evaluate(differentiate(differentiate(log(h), [a;b]), [a;b]),[a;b] => P)


gradDirectionalDeriv = zeros(ComplexF64, k)

for i = 1:k
    gradDirectionalDeriv[i] = sum((S.^-2).*SP[:,i])
end
gradDirectionalDeriv

# should be close to zero. And it is!!
norm(vec(transpose(B)*Hh) - gradDirectionalDeriv) 

grad = zeros(ComplexF64, k)

for i = 1:k
    grad[i] = sum((S.^-2).*SB[:,i])
end
grad
norm(Gh-grad) #also nearly zero.


# This is Jon's implementation of Hessian -- note that H*tranpose(SP UP) will not make sense for other k and n.
A = zeros(ComplexF64, length(S), size(F_on_line,1), k,k)
for i = 1:size(F_on_line,1)
    HF = differentiate(differentiate(F_on_line[i], vcat(t,u)), vcat(t,u))
    JxB = differentiate(differentiate(F_on_line[i], vcat(t,u)), β)
    JxP = differentiate(differentiate(F_on_line[i], vcat(t,u)), p)
    JBP = differentiate(differentiate(F_on_line[i], β), p)
    for j = 1:length(S)
        H = evaluate(HF, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        Jxb = evaluate(JxB, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        Jxp = evaluate(JxP, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        Jbp = evaluate(JBP, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))  
        A[j, i, :, :] = transpose(([SP[j,:] UP[j, :]]*H*transpose([SB[j,:] UB[j, :]])
                        + Jbp + [SP[j,:] UP[j,:]]*Jxb 
                        + transpose(Jxp)*transpose([SB[j, :] UB[j, :]]))) |> Matrix
        
    end
end


# Here is our attempt to get the mixed partials.
A = zeros(ComplexF64,length(F_on_line), length(S), k, k )
JsuF = differentiate(F_on_line, vcat(t, u))
for i in 1:length(F_on_line)
    JsuFβ = differentiate(differentiate(F_on_line[i], vcat(t, u)), β)
    JpFβ = differentiate(differentiate(F_on_line[i], p), β)
    for j in 1:length(S)
        subs_dict = Dict(
                    [p[k] => P[k] for k in eachindex(p)]...,
                    [β[k] => B[k] for k in eachindex(β)]...,
                    t => S[j],
                    [u[k] => U[:,j][k] for k in eachindex(u)]...
                )
    
        JsuFβ_eval = evaluate(JsuFβ, subs_dict)
        # display(JsuFβ_eval)
        JpFβ_eval = evaluate(JpFβ, subs_dict)
        # display(JpFβ_eval)
        #JsuF_eval = evaluate(JsuF, subs_dict) 
        ## I want to take the derivative of the following w.r.t β: J_(s,u)F ( ∇p(s) J_p u) + J_p F = 0
        # I think that this might be: (J_(s,u)F)_β(∇p(s) J_p u) + J_(s,u) ( ∇_b ∇_p s    (J_p u)_β)  + (J_p F)_β = 0
        # In order to take (J_(s,u)F)_β, I must do (J_(s,u)F[i])_β for each i and concatenate them. 
        # I store everything except J_(s,u) ( ∇_b ∇_p s    (J_p u)_β) in A, and will solve for ( ∇_b ∇_p s    (J_p u)_β) later
        
        A[i,j, :, :] = JsuFβ_eval * [SP[j,:] UP[j,:]] + JpFβ_eval 
        # i indexed by the number of equations
        # j indexed by number of intersections
    end
end

# The remaining plan is to concatenate all of A's along the first dimension, and then solve via 
# -JsuF \ A
# The first k x k block of the result will be ∇b∇pS, and the second k x k block will be ∇b∇pU.
A_collapsed = [hcat([A[j, i, :, :] for j in 1:size(A, 1)]...) for i in 1:size(A, 2)]



# We are solving for X in JsuF*X = -( JsuFβ*tranpose([SP JP]) + JpF\beta)
# X = ∇b∇pS = -JsuF \ (JsuFβ*transpose([SP; UP]) + JpFβ)
hess1 = zeros(ComplexF64, k, k)
hess2 = zeros(ComplexF64, k, k)
for j in 1:length(S)
    JsuF_eval = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
    soln = - JsuF_eval \ A_collapsed[j]
    ∇b∇pS = soln[1:k, 1:k]

    hess1 = hess1 - 2*S[j]^(-3)*SP[j,:]*transpose(SB[j,:])
    hess2 = hess2 + S[j]^(-2) * ∇b∇pS
end
hess = (hess1 + hess2 + Hlogqe(P))

disc = a^2 - 4 * b
actual_hess = hess_log_r_given_h(disc, e; c)
actual_hess(P)
hess

hess
Hh
show(A)
hess1 = zeros(ComplexF64, k, k)
hess2 = zeros(ComplexF64, k, k)
for j = 1:length(S)
    JtuInv = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))^(-1)
    Jtu = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
    A1 = Array(A[j,1,:,:])
    A2 = Array(A[j,2,:,:])
    sols =zeros(ComplexF64, k, k)
    for a in 1:k, b in 1:k
        rhs = [A1[a,b]; A2[a,b]]
        sols[a,b] = - (Jtu\ rhs)[1]
    end

    
    hess1 = hess1 - 2*S[j]^(-3)*SP[j,:]*transpose(SB[j,:])
    hess2 = hess2 + S[j]^(-2) *sols
end

hess = hess1 + hess2
norm(Hh - hess)
