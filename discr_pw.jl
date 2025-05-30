

is_real_vector = z -> all(abs.(imag.(z)) .< 1e-14)


####################################
####################################
####################################
####################################
####################################
####################################

n = 2 # ambient dimension
d = 2 # degree

# Variables
@var p[1:n] s[1:n, 1:d] u[1:n, 1:d]

# Parameters
@var Bv[1:n, 1:n]
#c = [1/3; 1/5; 1/7]
#c = 5 .* randn(n)



# Denominator
q = 1 + sum((p-c) .* (p-c))
∇q = differentiate(q, p)

# Intersection points should be on the discriminant   
evaluated_F = []
for i = 1:n
    for j = 1:d
        append!(evaluated_F, F([p + s[i, j] .* Bv[:, i]; u[i, j]]))
    end
end


#b_target = qr(randn(n,n)).Q |> Matrix




####################################
# with solve
####################################

# Gradient equations
# gradient_equations = map(1:n) do i
#     sum(prod(s[i,j] for j=1:d)/(-s[i,j]) for j=1:d)*q - e*transpose(∇q)*b[:,i]*prod(s[i,j] for j=1:d)
# end

# @var v[1:length(evaluated_F)], w[1:length(gradient_equations)]
# S = System(vcat(evaluated_F - v, gradient_equations - w), 
#                 variables=vcat(s[:],u[:],p), 
#                 parameters=vcat(b[:], v[:], w[:])) 

# vw0 = zeros(length(v) + length(w))

# solns = HC.solve(S, target_parameters =  vcat(B[:], vw0))

# k = length(s[:])
# true_values = filter(solutions(solns)) do sol
#     a = all(abs.(sol[1:k]) .> 1e-2)
#     b = abs(evaluate(q, p => sol[end-n+1:end])) > 1e-2
#     a && b
# end
# p_values = map(sol -> sol[end-n+1:end], true_values)
# real_p_values = filter(sol -> norm(imag(sol)) < 1e-14, p_values) |> real

#scatter!(Tuple.(real_p_values), markercolor = :green, label = "with solve", markersize = 8)

####################################
# with monodromy
####################################

#Gradient equations
gradient_equations = map(1:n) do i
    sum(1/(-s[i, j]) for j = 1:d) - e*transpose(∇q)*Bv[:, i]/q
end

@var V[1:length(evaluated_F)], W[1:length(gradient_equations)]
vw0 = zeros(length(V) + length(W))

S = System(
    vcat(evaluated_F - V, gradient_equations - W),
    variables = vcat(s[:], u[:], p),
    parameters = vcat(Bv[:], V[:], W[:]),
)

result = monodromy_solve(S)

# Specialize to our target parameters
solns = HC.solve(
    S,
    solutions(result),
    start_parameters = parameters(result),
    target_parameters = [vec(B); vw0],
)

p_values = map(sol -> sol[(end-n+1):end], solutions(solns))
real_p_values = filter(sol -> norm(imag(sol)) < 1e-14, p_values) |> real

real_p_values
