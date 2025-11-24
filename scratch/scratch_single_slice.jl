# An attempt to implement the single slice method for computing the Hessian

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


# To think about: Will length(S) be equal to d?
# To do: Use the pseudowitness set for the computation 
# (similar to what is done for the multiple slices!)

SP = zeros(ComplexF64, length(S), k)
SB = zeros(ComplexF64, length(S), k)
# Is the number of us always the same?
UP = zeros(ComplexF64, length(S), n-k, k)
UB = zeros(ComplexF64, length(S), n-k, k)
#UP = zeros(ComplexF64, length(S), k)
#UB = zeros(ComplexF64, length(S), k)
for i = 1:length(S)
    Jsu = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[i], U[:,i], P, B))
    JP = evaluate(JPF, vcat(t, u, p, β) => vcat(S[i], U[:,i], P, B))
    JB = evaluate(JBF, vcat(t, u, p, β) => vcat(S[i], U[:,i], P, B))
    A = -Jsu \ [JP JB] # solves the linear system Jsu * A = -[JP JB]
    display(A)
    SP[i,:] = A[1, 1:k]
    SB[i,:] = A[1, k+1:end]
    UP[i,:,:] = A[2:end, 1:k]
    UB[i,:,:] = A[2:end, k+1:end]
end



# TODO: Fix the above approach using PWS
all_vars = variables(F)
x_vars = setdiff(all_vars, projection_variables)
F_ordered = System(F.expressions, variables = [projection_variables; x_vars])
PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1)
GC = GradientCache(PWS)
track_pws_to_lines!(GC, P, B, PWS) # Our u's 

num_intersections = length(GC.line_hypersurface_intersections[bcol])
S = zeros(ComplexF64, num_intersections)
∇pS = zeros(ComplexF64, k, num_intersections)
for i = 1:num_intersections
    upoint = GC.line_hypersurface_intersections[bcol][i] # world coordinates of the intersection point prior to projection
    j = argmax(abs.(B[:, bcol]))
    S[i] = B[j, bcol] / (upoint[j] - P[j]) # The value of t that corresponds to the intersection point 
    subs_dict = Dict(
        [p[j] => P[j] for j in eachindex(p)]...,
        [b[j] => B[j, bcol] for j in eachindex(b)]...,
        t => S[i],
        [u[j] => upoint[k+j] for j in eachindex(u)]...,
    )
    JsuF_eval = evaluate(JsuF, subs_dict)
    JpF_eval = evaluate(JpF, subs_dict)
    sols = JsuF_eval \ (-JpF_eval) # This is a matrix [∇p s; Jp u]
    ∇pS[:, i] = sols[1, :]
end
########################    

#test: the SP we compute works
h = a^2-4*b
Gh = evaluate(differentiate(log(h), [a;b]), [a; b] => P )
Hh = evaluate(differentiate(differentiate(log(h), [a;b]), [a;b]),[a;b] => P)


gradDirectionalDeriv = sum(S[j]^(-2)*SP[j,:] for j=1:length(S))

# should be close to zero. And it is!!
norm(vec(transpose(B)*Hh) - gradDirectionalDeriv) 

#test: the SB we compute works
grad = zeros(ComplexF64, k)

# taking deriv w.r.t. b gives the s-derivative
grad = sum(S[j]^(-2)*SB[j,:] for j=1:length(S))

norm(Gh-grad) #also nearly zero.


# This is Jon's implementation of Hessian -- note that H*tranpose(SP UP) will not make sense for other k and n.
A = zeros(ComplexF64, length(S), size(F_on_line,1), k,k)
for i = 1:size(F_on_line,1)# loop through every equation in F?
    HF = differentiate(differentiate(F_on_line[i], vcat(t,u)), vcat(t,u))
    JxB = differentiate(differentiate(F_on_line[i], vcat(t,u)), β)
    JxP = differentiate(differentiate(F_on_line[i], vcat(t,u)), p)
    #JBP = differentiate(differentiate(F_on_line[i], β), p)
    JPB = differentiate(differentiate(F_on_line[i], p), β)
    for j = 1:length(S)
        H = evaluate(HF, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        Jxb = evaluate(JxB, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        Jxp = evaluate(JxP, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        #Jbp = evaluate(JBP, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))  
        Jpb = evaluate(JPB, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))  
        A[j, i, :, :] =  ([SP[j,:] transpose(UP[j, :, :])]*H*transpose([SB[j,:] transpose(UB[j, :, :])])
                        + Jpb + [SP[j,:] transpose(UP[j,:, :])]*Jxb 
                        + transpose(Jxp)*transpose([SB[j, :] transpose(UB[j, :, :])])) |> transpose |> Matrix
        
    end
end


# Here is our attempt to get the mixed partials.
A = zeros(ComplexF64,length(F_on_line), length(S), k, k )
JsuF = differentiate(F_on_line, vcat(t, u))
for i in 1:length(F_on_line) #n-k+1
    Fi = F_on_line[i]
    JsuFβ = differentiate(differentiate(Fi, vcat(t, u)), β)
    JpFβ = differentiate(differentiate(Fi, p), β)
    JsuFp = differentiate(differentiate(Fi, vcat(t,u)), p)
    hessF = differentiate(differentiate(Fi, vcat(t,u)),vcat(t,u))
    for j in 1:length(S) #d
        subs_dict = Dict(
                    [p[k] => P[k] for k in eachindex(p)]...,
                    [β[k] => B[k] for k in eachindex(β)]...,
                    t => S[j],
                    [u[k] => U[:,j][k] for k in eachindex(u)]...
                )
        
        JsuFβ_eval = evaluate(JsuFβ, subs_dict)
        # display(JsuFβ_eval)
        JpFβ_eval = evaluate(JpFβ, subs_dict)
        hessF_eval = evaluate(hessF, subs_dict)
        JsuFp_eval = evaluate(JsuFp, subs_dict)
        # display(JpFβ_eval)
        #JsuF_eval = evaluate(JsuF, subs_dict) 
        ## I want to take the derivative of the following w.r.t β: J_(s,u)F ( ∇p(s) J_p u) + J_p F = 0
        # I think that this might be: (J_(s,u)F)_β(∇p(s) J_p u) + J_(s,u) ( ∇_b ∇_p s    (J_p u)_β)  + (J_p F)_β = 0
        # In order to take (J_(s,u)F)_β, I must do (J_(s,u)F[i])_β for each i and concatenate them. 
        # I store everything except J_(s,u) ( ∇_b ∇_p s    (J_p u)_β) in A, and will solve for ( ∇_b ∇_p s    (J_p u)_β) later
        
        A[i,j, :, :] = hcat(SB[j,:],transpose(UB[j,:,:]))*hessF_eval*vcat(transpose(SP[j,:]), UP[j,:,:]) 
                    + JsuFβ_eval * vcat(transpose(SP[j,:]), UP[j,:,:]) + JsuFp_eval * vcat(transpose(SB[j,:]), UB[j,:,:])  + JpFβ_eval
        # i indexed by the number of equations
        # j indexed by number of intersections
    end
end

JsuF_eval = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B)) # this is length(F) x length(S)

Aj = hcat([A[j, i, :, :] for i in 1:size(A, 2)]...) #This concatenates each f in the system
A[1,1,:,:]
A[1,2,:,:]
soln = - JsuF_eval \ Aj

Fi = F_on_line[2]
JsuFp = differentiate(differentiate(Fi, vcat(t,u)), p)
# The remaining plan is to concatenate all of A's along the first dimension, and then solve via 
# -JsuF \ A
# The first k x k block of the result will be ∇b∇pS, and the second k x k block will be ∇b∇pU.
#A_collapsed = [hcat([A[j, i, :, :] for j in 1:size(A, 1)]...) for i in 1:size(A, 2)]
# j is index of equation f
# i is index of S
# size of A_collapsed is ?

# We are solving for X in JsuF*X = -( JsuFβ*tranpose([SP JP]) + JpF\beta)
# X = ∇b∇pS = -JsuF \ (JsuFβ*transpose([SP; UP]) + JpFβ)
hess1 = zeros(ComplexF64, k, k)
hess2 = zeros(ComplexF64, k, k)
for j in 1:length(S)
    Aj = hcat([A[i,j,:,:] for i in 1:length(F_on_line)]...) #This concatenates each f in the system 
    # size of Aj = (length F x length S) . (blah)
    JsuF_eval = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B)) # this is length(F) x length(S)
    # we expect the soln to be dimension blah x length(S)
    soln = - JsuF_eval \ Aj
    display(soln)
    ∇b∇pS = soln[1:k, 1:k]
    display(∇b∇pS)

    hess1 = hess1 - 2*S[j]^(-3)*SP[j,:]*transpose(SB[j,:])
    hess2 = hess2 + S[j]^(-2) * ∇b∇pS
end
hess = (hess1 + hess2 + Hlogqe(P))




disc = a^2 - 4 * b
actual_hess = hess_log_r_given_h(disc, e; c)
actual_hess(P)
Hh


i =1 
j=1 
Fi = F_on_line[i]
JsuFi = differentiate(Fi, vcat(t, u))
JsuFiβ = differentiate(JsuFi, β)
JpFiβ = transpose(differentiate(differentiate(Fi, p), β))

#JsuF_eval = evaluate(differentiate(F_on_line, vcat(t,u)), vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))

JsuFi_eval = evaluate(JsuFi, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
JpFiβ_eval = evaluate(JpFiβ, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
JsuFiβ_eval = evaluate(JsuFiβ, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))

A = JsuFiβ_eval*vcat(transpose(SP[j,:]), UP[j,:,:]) + JpFiβ_eval
anothersoln = - JsuFi_eval \ A

solns = zeros(ComplexF64, length(F_on_line), k, k)
for i in 1:length(F_on_line) #n-k+1
    Fi = F_on_line[i]
    JsuFi = differentiate(Fi, vcat(t, u))
    JsuFiβ = differentiate(JsuFi, β)
    JpFiβ = differentiate(differentiate(Fi, p), β)
    for j in 1:length(S)
        JsuFi_eval = evaluate(JsuFi, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        JpFiβ_eval = evaluate(JpFiβ, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        JsuFiβ_eval = evaluate(JsuFiβ, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        A = JsuFiβ_eval*transpose([SP[j,:] UP[j,:]]) + JpFiβ_eval
        soln = -JsuFi_eval \ A
    end
end

for j in 1:length(S)    
    for i in 1:length(F_on_line)
        Fi = F_on_line[i]
        JsuFi = transpose(differentiate(Fi, vcat(t, u)))
        JsuFiβ = differentiate(JsuFi, β)
        JpFiβ = differentiate(differentiate(Fi, p), β)



        JsuF_eval = evaluate(JsuF, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        JpFβ_eval = evaluate(JpFβ, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        JsuFβ_eval = evaluate(JsuFβ, vcat(t, u, p, β) => vcat(S[j], U[:,j], P, B))
        A = JsuFβ_eval*transpose([SP[j,:] UP[j,:]]) + JpFβ_eval
        soln = -JsuF_eval \ A
    end
end

differentiate(F_on_line[i], vcat(t, u[:]))

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
